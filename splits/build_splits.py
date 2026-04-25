"""Build train/eval/held-out-speaker splits from row_index.jsonl.

Outputs (all in splits/):
  speaker_to_rows.json       voice_id -> list of row indices into row_index.jsonl
  neutral_refs_by_speaker.json  voice_id -> list of row indices that have no inline tags
  locale_to_speakers.json    locale -> list of voice_ids
  train.json                 list of row indices for training
  eval_slices.json           structured eval set with tagged slices
  held_out_speakers.json     voice_ids withheld entirely from training (10%)

Eval slice design:
  S_tag_presence  – for each trained tag: a neutral row and a tagged row, same speaker, same locale
  S_cross_speaker – for each trained tag: rows from >=5 different speakers
  S_neutral       – tag-free rows from held-out speakers (baseline drift check)
  S_multi_tag     – rows with >=2 trained tags, held-out speakers

Run: .venv/bin/python splits/build_splits.py --analysis_dir analysis/ --splits_dir splits/
"""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_dir", default="analysis/")
    ap.add_argument("--splits_dir",   default="splits/")
    ap.add_argument("--held_out_frac", type=float, default=0.10,
                    help="Fraction of speakers held out entirely for eval")
    ap.add_argument("--eval_per_tag",  type=int, default=30,
                    help="Rows per tag for S_tag_presence slice (with_tag + without_tag)")
    ap.add_argument("--eval_cross_speaker_per_tag", type=int, default=5,
                    help="Min different speakers required for S_cross_speaker slice")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    ana = Path(args.analysis_dir)
    spl = Path(args.splits_dir)
    spl.mkdir(parents=True, exist_ok=True)

    # -- load whitelist --
    wl_path = spl / "tag_whitelist.json"
    with open(wl_path) as f:
        whitelist = json.load(f)
    trained_tags = set(
        whitelist["event"] + whitelist["prosody"] +
        whitelist["emotional"] + whitelist["neutral_baseline"]
    )
    print(f"Trained tags: {len(trained_tags)}")

    # -- load row index --
    print("Loading row_index.jsonl ...")
    rows = []
    with open(ana / "row_index.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"Total rows: {len(rows)}")

    # -- build speaker maps --
    speaker_to_rows:  dict[str, list[int]] = defaultdict(list)
    speaker_locale:   dict[str, str]       = {}
    locale_to_speakers: dict[str, list[str]] = defaultdict(list)
    neutral_refs:     dict[str, list[int]] = defaultdict(list)

    for idx, r in enumerate(rows):
        vid    = r["voice_id"]
        locale = r["locale"]
        speaker_to_rows[vid].append(idx)
        speaker_locale[vid] = locale
        locale_to_speakers[locale].append(vid)
        if not r["inline_tags"] and not r.get("tags_col"):
            neutral_refs[vid].append(idx)

    all_speakers = sorted(speaker_to_rows.keys())
    print(f"Unique speakers: {len(all_speakers)}")

    # deduplicate locale_to_speakers
    locale_to_speakers = {k: sorted(set(v)) for k, v in locale_to_speakers.items()}

    # -- held-out speakers (10%) --
    n_held = max(1, round(len(all_speakers) * args.held_out_frac))
    held_out = set(random.sample(all_speakers, n_held))
    train_speakers = [s for s in all_speakers if s not in held_out]
    print(f"Held-out speakers: {len(held_out)}, train speakers: {len(train_speakers)}")

    # -- build per-tag index for training speakers only --
    # tag -> list of (row_idx, voice_id) for training speakers
    tag_to_train_rows: dict[str, list[tuple[int,str]]] = defaultdict(list)
    for idx, r in enumerate(rows):
        if r["voice_id"] in held_out:
            continue
        for t in r["inline_tags"]:
            if t in trained_tags:
                tag_to_train_rows[t].append((idx, r["voice_id"]))

    # -- training split: all rows from train_speakers --
    train_indices = [idx for idx, r in enumerate(rows) if r["voice_id"] not in held_out]
    print(f"Train rows: {len(train_indices)}")

    # ----------------------------------------------------------------
    # Eval slices
    # ----------------------------------------------------------------
    eval_slices: dict[str, list[dict]] = {
        "S_tag_presence":    [],   # (with_tag, without_tag) pairs per tag
        "S_cross_speaker":   [],   # same tag across >=5 held-out speakers
        "S_neutral":         [],   # tag-free rows from held-out speakers
        "S_multi_tag":       [],   # >=2 trained tags, held-out speakers
    }

    # -- held-out speaker rows --
    held_rows: list[tuple[int, dict]] = [
        (idx, r) for idx, r in enumerate(rows) if r["voice_id"] in held_out
    ]

    # S_neutral
    neutral_held = [(idx, r) for idx, r in held_rows
                    if not r["inline_tags"] and not r.get("tags_col")]
    eval_slices["S_neutral"] = [
        {"row_idx": idx, "voice_id": r["voice_id"], "locale": r["locale"],
         "gender": r["gender"], "condition": "neutral"}
        for idx, r in random.sample(neutral_held, min(300, len(neutral_held)))
    ]

    # S_multi_tag
    multi_held = [(idx, r) for idx, r in held_rows
                  if len([t for t in r["inline_tags"] if t in trained_tags]) >= 2]
    eval_slices["S_multi_tag"] = [
        {"row_idx": idx, "voice_id": r["voice_id"], "locale": r["locale"],
         "tags": [t for t in r["inline_tags"] if t in trained_tags]}
        for idx, r in random.sample(multi_held, min(200, len(multi_held)))
    ]

    # S_tag_presence and S_cross_speaker: per trained tag
    held_by_speaker: dict[str, list[tuple[int,dict]]] = defaultdict(list)
    for idx, r in held_rows:
        held_by_speaker[r["voice_id"]].append((idx, r))

    for tag in trained_tags:
        if tag == "normal voice":
            continue

        # rows from held-out speakers that HAVE this tag
        with_tag = [(idx, r) for idx, r in held_rows
                    if tag in r["inline_tags"]]
        # sample one per speaker to avoid clustering
        by_spk: dict[str, list] = defaultdict(list)
        for idx, r in with_tag:
            by_spk[r["voice_id"]].append((idx, r))
        sampled_with = []
        for spk, items in by_spk.items():
            sampled_with.append(random.choice(items))
        sampled_with = random.sample(sampled_with, min(args.eval_per_tag, len(sampled_with)))

        # matching without-tag rows from the same speakers, same locale
        sampled_without = []
        for idx, r in sampled_with:
            spk = r["voice_id"]
            candidates = neutral_refs.get(spk, [])
            # also accept rows from the same speaker with no trained tags
            if not candidates:
                candidates = [i for i, rr in enumerate(rows)
                               if rr["voice_id"] == spk and not any(t in trained_tags for t in rr["inline_tags"])]
            if candidates:
                sampled_without.append(random.choice(candidates))
            else:
                sampled_without.append(None)

        for i, (idx_w, r_w) in enumerate(sampled_with):
            entry = {
                "tag": tag, "category": _tag_category(tag, whitelist),
                "with_tag_row": idx_w, "voice_id": r_w["voice_id"],
                "locale": r_w["locale"],
                "without_tag_row": sampled_without[i],
            }
            eval_slices["S_tag_presence"].append(entry)

        # S_cross_speaker: collect across as many held-out speakers as possible
        cross = [{"row_idx": idx, "voice_id": r["voice_id"], "locale": r["locale"], "tag": tag}
                 for idx, r in sampled_with]
        if len(set(e["voice_id"] for e in cross)) >= args.eval_cross_speaker_per_tag:
            eval_slices["S_cross_speaker"].extend(cross)

    print(f"Eval S_tag_presence:  {len(eval_slices['S_tag_presence'])} entries")
    print(f"Eval S_cross_speaker: {len(eval_slices['S_cross_speaker'])} entries")
    print(f"Eval S_neutral:       {len(eval_slices['S_neutral'])} entries")
    print(f"Eval S_multi_tag:     {len(eval_slices['S_multi_tag'])} entries")

    # ----------------------------------------------------------------
    # Write outputs
    # ----------------------------------------------------------------
    with open(spl / "speaker_to_rows.json", "w") as f:
        json.dump(dict(speaker_to_rows), f)

    with open(spl / "neutral_refs_by_speaker.json", "w") as f:
        json.dump(dict(neutral_refs), f)

    with open(spl / "locale_to_speakers.json", "w") as f:
        json.dump(dict(locale_to_speakers), f)

    with open(spl / "held_out_speakers.json", "w") as f:
        json.dump(sorted(held_out), f, indent=2)

    with open(spl / "train.json", "w") as f:
        json.dump(train_indices, f)

    with open(spl / "eval_slices.json", "w") as f:
        json.dump(eval_slices, f, indent=2)

    print(f"\nAll split files written to {spl}/")
    print(f"  train.json:             {len(train_indices)} rows")
    print(f"  held_out_speakers.json: {len(held_out)} speakers")
    print(f"  eval_slices.json:       slices: {list(eval_slices.keys())}")


def _tag_category(tag: str, whitelist: dict) -> str:
    for cat in ("event", "prosody", "emotional", "neutral_baseline"):
        if tag in whitelist.get(cat, []):
            return cat
    return "other"


if __name__ == "__main__":
    main()
