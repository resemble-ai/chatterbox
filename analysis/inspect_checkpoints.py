"""Diagnose how far each finetune checkpoint has drifted from the pretrained Turbo T3 weights.

For every checkpoint in --ckpt_dir, computes the L2 distance and relative drift
(‖θ_ckpt − θ_pretrained‖₂ / ‖θ_pretrained‖₂) per module group.  A sudden jump
between consecutive checkpoints is the signature of training divergence.

Module groups inspected:
    text_emb     — input text embedding table  (50276+N × 1024)
    text_head    — text output projection
    speech_emb   — speech token input embedding (6563 × 1024)
    speech_head  — speech output projection (the main loss head)
    cond_enc     — T3CondEnc (speaker / prompt / emotion conditioning)
    tfmr         — GPT-2 medium backbone layers
    other        — everything else (positional embs, layer norms, etc.)

Also parses any `training_*.log` files in the checkpoint dir and dumps the loss
curve so you can correlate divergence with the loss trajectory.

Usage
-----
.venv/bin/python analysis/inspect_checkpoints.py --ckpt_dir checkpoints/
.venv/bin/python analysis/inspect_checkpoints.py --ckpt_dir checkpoints/ --csv /tmp/drift.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ─────────────────────────────────────────────────────────────────────────────
# Module-group classifier
# ─────────────────────────────────────────────────────────────────────────────

GROUP_PATTERNS = [
    ("text_emb",    re.compile(r"^text_emb\.")),
    ("text_head",   re.compile(r"^text_head\.")),
    ("speech_emb",  re.compile(r"^speech_emb\.")),
    ("speech_head", re.compile(r"^speech_head\.")),
    ("cond_enc",    re.compile(r"^cond_enc\.")),
    ("tfmr",        re.compile(r"^tfmr\.")),
]

def classify(key: str) -> str:
    for name, pat in GROUP_PATTERNS:
        if pat.match(key):
            return name
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# Drift calculation
# ─────────────────────────────────────────────────────────────────────────────

def per_group_norms(state: dict[str, torch.Tensor]) -> dict[str, float]:
    """Return ‖θ‖₂ per group."""
    sums = defaultdict(float)
    for k, v in state.items():
        if not torch.is_floating_point(v):
            continue
        g = classify(k)
        sums[g] += float(v.detach().to(torch.float32).pow(2).sum().item())
    return {g: float(s) ** 0.5 for g, s in sums.items()}


def per_group_drift(pre: dict, ckpt: dict, expected_text_vocab: int | None) -> tuple[dict[str, float], dict[str, float]]:
    """Return (absolute_drift, relative_drift) per group.

    The pretrained text_emb has 50276 rows; the checkpoint has 50276+N. We compare
    ONLY the overlapping rows (the first 50276) for fair drift; the new rows have
    no pretrained counterpart.
    """
    abs_drift = defaultdict(float)
    norm_pre  = defaultdict(float)
    extra_keys = []
    missing_keys = []

    for k, v_post in ckpt.items():
        if k not in pre:
            extra_keys.append(k)
            continue
        if not torch.is_floating_point(v_post):
            continue
        v_pre = pre[k]

        # Handle vocab-expanded layers: align shapes
        if v_post.shape != v_pre.shape:
            if k == "text_emb.weight" or k == "text_head.weight":
                # take the overlap (first N_pre rows / cols)
                if v_pre.dim() == 2 and v_post.dim() == 2:
                    n = min(v_pre.shape[0], v_post.shape[0])
                    v_post_overlap = v_post[:n]
                    v_pre_overlap  = v_pre[:n]
                else:
                    continue
                diff = (v_post_overlap.to(torch.float32) - v_pre_overlap.to(torch.float32)).pow(2).sum().item()
                pre_sq = v_pre_overlap.to(torch.float32).pow(2).sum().item()
            else:
                continue  # shape mismatch elsewhere — skip
        else:
            diff   = (v_post.to(torch.float32) - v_pre.to(torch.float32)).pow(2).sum().item()
            pre_sq = v_pre.to(torch.float32).pow(2).sum().item()

        g = classify(k)
        abs_drift[g] += diff
        norm_pre[g]  += pre_sq

    for k in pre:
        if k not in ckpt:
            missing_keys.append(k)

    abs_drift = {g: float(s) ** 0.5 for g, s in abs_drift.items()}
    rel_drift = {g: (abs_drift[g] / float(norm_pre[g]) ** 0.5) if norm_pre[g] > 0 else 0.0
                 for g in abs_drift}
    return abs_drift, rel_drift


# ─────────────────────────────────────────────────────────────────────────────
# Log file parsing
# ─────────────────────────────────────────────────────────────────────────────

LOG_LINE_RE = re.compile(
    r"step=\s*(\d+)\s+loss=([\d.eE+\-nan]+)\s+\(text=([\d.eE+\-nan]+)\s+speech=([\d.eE+\-nan]+)\)"
)

def parse_logs(ckpt_dir: Path) -> list[dict]:
    out = []
    for log_path in sorted(ckpt_dir.glob("training_*.log")):
        with open(log_path) as f:
            for line in f:
                m = LOG_LINE_RE.search(line)
                if m:
                    step, loss, lt, ls = m.groups()
                    out.append({
                        "log": log_path.name,
                        "step": int(step),
                        "loss": float(loss),
                        "loss_text":   float(lt),
                        "loss_speech": float(ls),
                    })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="checkpoints/")
    ap.add_argument("--csv",      default=None,
                    help="If set, write per-checkpoint drift to this CSV.")
    ap.add_argument("--device",   default="cpu",
                    help="cpu (default) is plenty — the math is just norms.")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    if not ckpts:
        print(f"No step_*.pt files in {ckpt_dir}/")
        return

    # ── load pretrained Turbo T3 once ─────────────────────────────────────
    print("[inspect] loading pretrained Chatterbox-Turbo T3 ...", flush=True)
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    model = ChatterboxTurboTTS.from_pretrained(device=args.device)
    pre_state = {k: v.detach().cpu() for k, v in model.t3.state_dict().items()}
    pre_norms = per_group_norms(pre_state)
    print(f"[inspect] pretrained ‖θ‖ per group:")
    for g, n in sorted(pre_norms.items()):
        print(f"    {g:<12} {n:>14.4f}")
    del model
    print()

    # ── parse logs ────────────────────────────────────────────────────────
    log_rows = parse_logs(ckpt_dir)
    if log_rows:
        print(f"[inspect] parsed {len(log_rows)} log entries from training_*.log files")
        # show the loss curve at the same steps as our ckpts
        ckpt_steps = {int(re.search(r"step_(\d+)\.pt", p.name).group(1)) for p in ckpts}
        relevant = [r for r in log_rows if r["step"] in ckpt_steps]
        if relevant:
            print(f"{'step':>7} {'loss':>10} {'text':>10} {'speech':>10}")
            for r in sorted(relevant, key=lambda x: x["step"]):
                print(f"{r['step']:>7} {r['loss']:>10.4f} {r['loss_text']:>10.4f} {r['loss_speech']:>10.4f}")
            print()
    else:
        print("[inspect] no training_*.log files found in ckpt dir — only weight diagnostics will run")
        print()

    # ── drift per checkpoint ──────────────────────────────────────────────
    rows = []
    print(f"[inspect] drift per checkpoint (relative = ‖θ_ckpt − θ_pre‖ / ‖θ_pre‖):")
    groups = ["text_emb", "text_head", "speech_emb", "speech_head", "cond_enc", "tfmr", "other"]
    print(f"{'step':>7}  " + " ".join(f"{g:>11}" for g in groups) + f"  {'TOTAL_rel':>10}")
    last_total = None
    for cp in ckpts:
        m = re.search(r"step_(\d+)\.pt", cp.name)
        step = int(m.group(1))
        ckpt = torch.load(cp, map_location=args.device, weights_only=False)
        t3_state = ckpt.get("t3_state", ckpt)
        abs_d, rel_d = per_group_drift(pre_state, t3_state, ckpt.get("text_vocab_size"))
        # total relative drift over all groups
        all_pre_sq  = sum(pre_norms[g] ** 2 for g in pre_norms)
        all_diff_sq = sum(abs_d.get(g, 0.0) ** 2 for g in pre_norms)
        total_rel   = (all_diff_sq ** 0.5) / (all_pre_sq ** 0.5) if all_pre_sq > 0 else 0.0
        spike = ""
        if last_total is not None and total_rel > 1.5 * last_total:
            spike = "  ← SPIKE"
        last_total = total_rel
        line = f"{step:>7}  " + " ".join(f"{rel_d.get(g, 0.0):>11.4f}" for g in groups) + f"  {total_rel:>10.4f}{spike}"
        print(line)
        row = {"step": step, "total_rel_drift": total_rel}
        for g in groups:
            row[f"abs_{g}"] = abs_d.get(g, 0.0)
            row[f"rel_{g}"] = rel_d.get(g, 0.0)
        rows.append(row)

    if args.csv:
        out_path = Path(args.csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r.keys()})
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n[inspect] wrote {out_path}")

    print()
    print("Reading the table:")
    print("  - rel_drift values are unitless. 0.01 = the layer moved 1% relative to its pretrained norm.")
    print("  - Healthy finetuning: total_rel_drift grows smoothly, often plateauing at a few percent.")
    print("  - DIVERGENCE: any group jumping > 50% (especially text_head, speech_head, or tfmr) between")
    print("    consecutive checkpoints means the optimizer overshot. The first SPIKE marker is the prime")
    print("    suspect for when generation broke.")


if __name__ == "__main__":
    main()
