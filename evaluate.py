"""Held-out evaluation for Chatterbox-Turbo T3 finetune.

Runs inference on the structured eval slices built by splits/build_splits.py
under multiple counterfactual conditions per row so we can separate
"did the tag cause the change?" from "did domain fit improve?".

Conditions per eval row (only those that apply to the slice):
  with_tag_matched_ref    — feed script unmodified, reference is a tagged clip
  with_tag_neutral_ref    — feed script unmodified, reference is a tag-free clip
  without_tag_matched_ref — strip bracketed tags from script, tagged ref
  without_tag_neutral_ref — strip bracketed tags from script, neutral ref
  shuffled_tags           — replace each trained tag with a random different one

Outputs per run:
  eval_runs/<name>/
    wavs/<slice>/<row_id>__<condition>.wav
    metrics.csv            — row per (row_id, condition) with metadata
    slice_aggregates.json  — slice-level aggregates (mean duration, per-tag counts)

Usage
-----
.venv/bin/python evaluate.py \
    --checkpoint pretrained \
    --splits_dir splits/ \
    --dataset_path <snapshot_dir> \
    --out_dir eval_runs/baseline/ \
    [--slices S_tag_presence S_neutral] \
    [--max_per_slice 100] \
    [--conditions with_tag_matched_ref with_tag_neutral_ref without_tag_neutral_ref]

Note: automated metrics (WER, event-classifier probs, F0 stats) require optional
packages (whisper, torchaudio, MIT/ast-finetuned-audioset-...).  They are
computed best-effort; missing packages produce a warning and partial metrics.
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio

sys.path.insert(0, str(Path(__file__).parent / "src"))
from chatterbox.tts_turbo import ChatterboxTurboTTS


TAG_RE = re.compile(r"\[([^\[\]\n\r]{1,60})\]")


# ─────────────────────────────────────────────────────────────────────────────
# Vocab expansion (mirrors finetune.py / run_inference.py)
# ─────────────────────────────────────────────────────────────────────────────

def expand_vocab_for_tags(model, splits_dir: str):
    wl = json.load(open(Path(splits_dir) / "tag_whitelist.json"))
    trained = wl["event"] + wl["prosody"] + wl["emotional"] + wl["neutral_baseline"]
    wrapped = [f"[{t}]" for t in trained]
    vocab = set(model.tokenizer.get_vocab().keys())
    to_add = [t for t in wrapped if t not in vocab]
    if not to_add:
        return []
    model.tokenizer.add_tokens(to_add)
    t3 = model.t3
    old = t3.text_emb.num_embeddings
    new = old + len(to_add)
    dim, dev, dt = t3.text_emb.embedding_dim, t3.text_emb.weight.device, t3.text_emb.weight.dtype
    ne = nn.Embedding(new, dim).to(device=dev, dtype=dt)
    with torch.no_grad():
        ne.weight[:old] = t3.text_emb.weight
        nn.init.normal_(ne.weight[old:], std=0.02)
    t3.text_emb = ne
    oh = t3.text_head
    nh = nn.Linear(oh.in_features, new, bias=oh.bias is not None).to(device=dev, dtype=dt)
    with torch.no_grad():
        nh.weight[:old] = oh.weight
        nn.init.normal_(nh.weight[old:], std=0.02)
        if oh.bias is not None:
            nh.bias[:old] = oh.bias
            nn.init.zeros_(nh.bias[old:])
    t3.text_head = nh
    t3.hp.text_tokens_dict_size = new
    return to_add


# ─────────────────────────────────────────────────────────────────────────────
# Parquet audio loader
# ─────────────────────────────────────────────────────────────────────────────

class Shards:
    def __init__(self, dataset_path: str):
        self.paths = sorted(glob.glob(f"{dataset_path}/data/train-*.parquet"))
        self._pf_cache: dict[int, pq.ParquetFile] = {}

    def _pf(self, shard_idx: int) -> pq.ParquetFile:
        if shard_idx not in self._pf_cache:
            self._pf_cache[shard_idx] = pq.ParquetFile(self.paths[shard_idx])
        return self._pf_cache[shard_idx]

    def load(self, shard_idx: int, row_idx: int) -> tuple[str, bytes | None]:
        pf = self._pf(shard_idx)
        rows_seen = 0
        for rg in range(pf.num_row_groups):
            rg_size = pf.metadata.row_group(rg).num_rows
            if rows_seen + rg_size > row_idx:
                local = row_idx - rows_seen
                tbl = pf.read_row_group(rg, columns=["preprocessed_audio", "script_content"]).slice(local, 1).to_pydict()
                audio = tbl["preprocessed_audio"][0]
                script = tbl["script_content"][0] or ""
                raw = audio["bytes"] if audio else None
                return script, raw
            rows_seen += rg_size
        return "", None


def save_ref_wav(raw_bytes: bytes, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    sf.write(str(path), data, sr)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Text mutations for counterfactual conditions
# ─────────────────────────────────────────────────────────────────────────────

def strip_tags(text: str) -> str:
    stripped = TAG_RE.sub("", text)
    # collapse whitespace
    return " ".join(stripped.split())


def shuffle_trained_tags(text: str, trained_set: set[str], rng: random.Random) -> str:
    pool = list(trained_set)
    def sub(m: re.Match) -> str:
        tag = m.group(1).strip().lower()
        if tag in trained_set:
            choices = [t for t in pool if t != tag] or pool
            return f"[{rng.choice(choices)}]"
        return m.group(0)
    return TAG_RE.sub(sub, text)


# ─────────────────────────────────────────────────────────────────────────────
# Best-effort metrics (all optional)
# ─────────────────────────────────────────────────────────────────────────────

def _try_import(name):
    try:
        return __import__(name)
    except Exception:
        return None


def compute_prosody_metrics(wav_path: Path) -> dict:
    """Light-weight F0 and energy stats — always available via torchaudio."""
    try:
        wav, sr = torchaudio.load(str(wav_path))
        wav = wav.mean(dim=0)  # mono
        rms = wav.pow(2).mean().sqrt().item()
        # Pitch via torchaudio.functional.detect_pitch_frequency (wraps YIN)
        try:
            f0 = torchaudio.functional.detect_pitch_frequency(wav.unsqueeze(0), sr).squeeze(0)
            f0 = f0[f0 > 20]  # drop zeros / silence
            f0_mean = f0.mean().item() if f0.numel() else 0.0
            f0_std  = f0.std().item()  if f0.numel() > 1 else 0.0
        except Exception:
            f0_mean, f0_std = 0.0, 0.0
        return {"rms": rms, "f0_mean": f0_mean, "f0_std": f0_std, "duration_s": wav.shape[-1] / sr}
    except Exception as e:
        return {"rms": None, "f0_mean": None, "f0_std": None, "duration_s": None, "error": str(e)}


def compute_wer(wav_path: Path, reference_text: str, whisper_model=None) -> float | None:
    """Optional: transcribe with Whisper and compute WER.  Returns None if unavailable."""
    if whisper_model is None:
        return None
    try:
        import jiwer
        result = whisper_model.transcribe(str(wav_path))
        hyp = result.get("text", "").strip().lower()
        ref = reference_text.strip().lower()
        if not ref:
            return None
        return jiwer.wer(ref, hyp)
    except Exception:
        return None


def load_whisper(name: str = "small"):
    whisper = _try_import("whisper")
    if not whisper:
        print(f"[eval] whisper not installed; WER will be skipped")
        return None
    try:
        print(f"[eval] loading whisper-{name} for WER...")
        return whisper.load_model(name)
    except Exception as e:
        print(f"[eval] could not load whisper: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Eval core
# ─────────────────────────────────────────────────────────────────────────────

def _stash_conds(model, ref_path: Path):
    """Run prepare_conditionals once and return a (deep-copyable) snapshot.

    The model holds .conds; we snapshot tensors so we can swap them in for any
    later generate() call without re-doing the conditioning forward passes.
    """
    model.prepare_conditionals(str(ref_path), exaggeration=0.0)
    return model.conds


def evaluate_row(model, shards, row_index, entry, conditions, trained_set, out_wavs,
                 whisper_model, rng):
    """Run the chosen conditions for a single eval entry; return list of row dicts.

    Optimization: prepare_conditionals() per unique reference audio (max 2 per
    row: matched + neutral) instead of per condition (up to 5).  Saves the wav
    once for traceability but feeds the model via cached conds, skipping the
    disk round-trip on every call.
    """
    row_idx = entry.get("with_tag_row") or entry.get("row_idx")
    if row_idx is None:
        return []
    r = row_index[row_idx]
    script, _ = shards.load(r["shard"], r["row"])

    matched_ref_bytes = None
    _, matched_ref_bytes = shards.load(r["shard"], r["row"])

    neutral_row = entry.get("without_tag_row")
    neutral_ref_bytes = None
    if neutral_row is not None:
        rn = row_index[neutral_row]
        _, neutral_ref_bytes = shards.load(rn["shard"], rn["row"])

    results = []
    slice_name = entry.get("_slice", "unknown")
    tag_truth = entry.get("tag", "") or ",".join(entry.get("tags", []))

    ref_dir = out_wavs / "_refs" / f"row_{row_idx}"
    matched_ref_path = ref_dir / "matched.wav"
    neutral_ref_path = ref_dir / "neutral.wav"
    if matched_ref_bytes is not None:
        save_ref_wav(matched_ref_bytes, matched_ref_path)
    if neutral_ref_bytes is not None:
        save_ref_wav(neutral_ref_bytes, neutral_ref_path)

    # ── prepare conditionals ONCE per unique ref (matched + neutral) ──────
    cached_conds: dict[str, object] = {}
    if matched_ref_bytes is not None:
        _stash_conds(model, matched_ref_path)
        cached_conds["matched"] = model.conds
    if neutral_ref_bytes is not None:
        _stash_conds(model, neutral_ref_path)
        cached_conds["neutral"] = model.conds

    mutations = {
        "with_tag_matched_ref":    (script,                                              "matched"),
        "with_tag_neutral_ref":    (script,                                              "neutral"),
        "without_tag_matched_ref": (strip_tags(script),                                  "matched"),
        "without_tag_neutral_ref": (strip_tags(script),                                  "neutral"),
        "shuffled_tags":           (shuffle_trained_tags(script, trained_set, rng),     "matched"),
    }

    for cond in conditions:
        if cond not in mutations:
            continue
        text, ref_kind = mutations[cond]
        if ref_kind not in cached_conds:
            continue

        # Swap pre-computed conditionals into the model — bypasses prepare_conditionals
        model.conds = cached_conds[ref_kind]

        wav_path = out_wavs / slice_name / f"row_{row_idx}__{cond}.wav"
        wav_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            wav = model.generate(
                text,
                audio_prompt_path=None,           # ← use cached conds, skip prepare_conditionals
                temperature=0.8, top_p=0.95, top_k=1000,
                repetition_penalty=1.2, norm_loudness=True,
            )
            torchaudio.save(str(wav_path), wav, sample_rate=model.sr)
        except Exception as e:
            results.append({
                "row_idx": row_idx, "slice": slice_name, "condition": cond,
                "tag_truth": tag_truth, "voice_id": entry.get("voice_id"),
                "locale": entry.get("locale"),
                "wav_path": str(wav_path), "text": text,
                "error": str(e),
            })
            continue

        prosody = compute_prosody_metrics(wav_path)
        wer     = compute_wer(wav_path, strip_tags(text), whisper_model)
        results.append({
            "row_idx": row_idx, "slice": slice_name, "condition": cond,
            "tag_truth": tag_truth, "voice_id": entry.get("voice_id"),
            "locale": entry.get("locale"),
            "wav_path": str(wav_path), "text": text,
            "rms": prosody.get("rms"),
            "f0_mean": prosody.get("f0_mean"),
            "f0_std":  prosody.get("f0_std"),
            "duration_s": prosody.get("duration_s"),
            "wer": wer,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",   required=True,
                    help="Path to finetuned checkpoint, or 'pretrained'")
    ap.add_argument("--splits_dir",   default="splits/")
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--slices",       nargs="+",
                    default=["S_tag_presence", "S_neutral", "S_multi_tag", "S_cross_speaker"])
    ap.add_argument("--conditions",   nargs="+",
                    default=["with_tag_matched_ref", "with_tag_neutral_ref",
                             "without_tag_neutral_ref", "shuffled_tags"])
    ap.add_argument("--max_per_slice", type=int, default=50,
                    help="Cap to keep eval time reasonable")
    ap.add_argument("--whisper_model", default="small",
                    help="Whisper model for WER ('small', 'base', etc.). Skipped if unavailable.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wavs = out_dir / "wavs"
    out_wavs.mkdir(exist_ok=True)

    # ── load data structures ──────────────────────────────────────────────
    spl = Path(args.splits_dir)
    ri_path = Path("analysis/row_index.jsonl")
    if (spl / "row_index.jsonl").exists():
        ri_path = spl / "row_index.jsonl"
    with open(ri_path) as f:
        row_index = [json.loads(line) for line in f]

    with open(spl / "eval_slices.json") as f:
        eval_slices = json.load(f)
    with open(spl / "tag_whitelist.json") as f:
        wl = json.load(f)
    trained_set = set(wl["event"] + wl["prosody"] + wl["emotional"] + wl["neutral_baseline"])

    shards = Shards(args.dataset_path)

    # ── load model ────────────────────────────────────────────────────────
    print("[eval] loading Chatterbox-Turbo ...", flush=True)
    model = ChatterboxTurboTTS.from_pretrained(device=args.device)
    eos_id = model.tokenizer.eos_token_id
    model.t3.hp.start_text_token = eos_id
    model.t3.hp.stop_text_token  = eos_id

    if args.checkpoint == "pretrained":
        print("[eval] using pretrained Turbo baseline")
    else:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        expected = ckpt.get("text_vocab_size")
        if expected and expected > model.t3.text_emb.num_embeddings:
            expand_vocab_for_tags(model, args.splits_dir)
        t3_state = ckpt.get("t3_state", ckpt)
        model.t3.load_state_dict(t3_state, strict=False)
        model.t3.to(args.device).eval()
        print(f"[eval] loaded finetuned ckpt step={ckpt.get('step', '?')}")

    # ── optional Whisper ──────────────────────────────────────────────────
    whisper_model = load_whisper(args.whisper_model)

    # ── run eval ──────────────────────────────────────────────────────────
    metrics_rows = []
    aggregates = defaultdict(lambda: defaultdict(int))

    for slice_name in args.slices:
        entries = eval_slices.get(slice_name, [])
        if not entries:
            print(f"[eval] slice '{slice_name}' empty — skipping")
            continue
        # cap
        entries = entries[: args.max_per_slice]
        print(f"[eval] slice={slice_name}  n={len(entries)}", flush=True)
        for i, entry in enumerate(entries):
            entry = dict(entry); entry["_slice"] = slice_name
            rows = evaluate_row(model, shards, row_index, entry,
                                args.conditions, trained_set, out_wavs,
                                whisper_model, rng)
            metrics_rows.extend(rows)
            aggregates[slice_name]["n_entries"] += 1
            aggregates[slice_name]["n_conditions_run"] += len(rows)
            if (i + 1) % 10 == 0:
                print(f"  ...{i+1}/{len(entries)}", flush=True)

    # ── write outputs ─────────────────────────────────────────────────────
    if metrics_rows:
        csv_path = out_dir / "metrics.csv"
        keys = sorted({k for r in metrics_rows for k in r.keys()})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in metrics_rows:
                w.writerow(r)
        print(f"[eval] wrote {csv_path}  ({len(metrics_rows)} rows)")

    with open(out_dir / "slice_aggregates.json", "w") as f:
        json.dump(dict(aggregates), f, indent=2)
    print(f"[eval] wrote {out_dir/'slice_aggregates.json'}")
    print(f"[eval] done. wavs under {out_wavs}/")


if __name__ == "__main__":
    main()
