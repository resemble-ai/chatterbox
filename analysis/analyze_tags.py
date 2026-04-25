"""Phase 0 dataset analysis for paralinguistic-tag finetuning.

Streams the parquet shards of the stage1 dataset and produces:
  - tag_counts.csv          tag -> row count, speaker count, mean duration
  - tag_cooccurrence.csv    pairs of tags appearing in the same row
  - tag_per_speaker.csv     (speaker, tag, count)
  - neutral_rows_per_speaker.csv
  - tags_vs_tags_field.json precision/recall of inline bracket extraction vs the `tags` column
  - tag_tokenization.json   EnTokenizer output for each tag wrapped as '[tag]'
  - top40.txt               human-readable summary printed to stdout too

Run: .venv/bin/python analysis/analyze_tags.py --dataset_path <snap> --out_dir analysis/
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

# chatterbox tokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chatterbox.models.tokenizers.tokenizer import EnTokenizer, UNK

TAG_RE = re.compile(r"\[([^\[\]\n\r]{1,60})\]")
SPEAKER_RE = re.compile(r"voices/([0-9a-fA-F\-]{8,})/")


def extract_tags(text: str) -> set[str]:
    if not text:
        return set()
    return {m.group(1).strip().lower() for m in TAG_RE.finditer(text) if m.group(1).strip()}


def extract_speaker(gcs_path: str | None) -> str | None:
    if not gcs_path:
        return None
    m = SPEAKER_RE.search(gcs_path)
    return m.group(1) if m else None


def iter_rows(shards):
    cols = ["script_content", "tags", "gender", "gcs_path", "duration_seconds"]
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=cols)
            data = tbl.to_pydict()
            n = len(data["script_content"])
            for i in range(n):
                yield {
                    "script_content": data["script_content"][i],
                    "tags": data["tags"][i] or [],
                    "gender": data["gender"][i],
                    "gcs_path": data["gcs_path"][i],
                    "duration_seconds": data["duration_seconds"][i],
                }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True,
                    help="Path to snapshot dir (contains data/train-*.parquet)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_k_for_tokenization", type=int, default=200)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(glob.glob(str(Path(args.dataset_path) / "data" / "train-*.parquet")))
    if not shards:
        sys.exit(f"No shards under {args.dataset_path}/data/")
    print(f"[analyze_tags] scanning {len(shards)} shards", flush=True)

    # ---- stream ----
    tag_rows = Counter()               # tag -> n_rows containing it
    tag_speakers = defaultdict(set)    # tag -> set(speaker_id)
    tag_durations = defaultdict(list)  # tag -> list[float] durations (mean only)
    cooc = Counter()                   # frozenset({a,b}) -> n_rows
    per_speaker = Counter()            # (speaker_id, tag) -> count
    total_rows = 0
    neutral_rows_per_speaker = Counter()
    # cross-check between the `tags` column and the inline bracket regex
    tagcol_seen = Counter()            # lowercased tags-field values
    inline_seen_for_tagcol_key = Counter()  # how often the same string appears inline too

    # row index for later split build (shard_idx, row_idx_in_shard)
    row_index = []   # [(shard_idx, row_idx_in_shard, speaker_id, duration, inline_tags_list, tags_col_list)]

    for shard_idx, shard in enumerate(shards):
        pf = pq.ParquetFile(shard)
        row_idx = 0
        cols = ["script_content", "tags", "gcs_path", "duration_seconds"]
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=cols)
            data = tbl.to_pydict()
            n = len(data["script_content"])
            for i in range(n):
                sc = data["script_content"][i] or ""
                tc = [t.lower().strip() for t in (data["tags"][i] or []) if t and t.strip()]
                gcs = data["gcs_path"][i]
                dur = data["duration_seconds"][i] or 0.0
                inline = extract_tags(sc)
                speaker = extract_speaker(gcs) or "UNKNOWN"

                # row index entry
                row_index.append((shard_idx, row_idx, speaker, float(dur),
                                  sorted(inline), sorted(set(tc))))

                # aggregates
                if inline:
                    for t in inline:
                        tag_rows[t] += 1
                        tag_speakers[t].add(speaker)
                        tag_durations[t].append(float(dur))
                        per_speaker[(speaker, t)] += 1
                    for a in inline:
                        for b in inline:
                            if a < b:
                                cooc[(a, b)] += 1
                if not inline and not tc:
                    neutral_rows_per_speaker[speaker] += 1

                # tags-field cross-check
                for t in tc:
                    tagcol_seen[t] += 1
                    if t in inline:
                        inline_seen_for_tagcol_key[t] += 1

                total_rows += 1
                row_idx += 1
        if (shard_idx + 1) % 10 == 0:
            print(f"  ...{shard_idx+1}/{len(shards)} shards  rows={total_rows}", flush=True)

    print(f"[analyze_tags] total_rows={total_rows}, unique_inline_tags={len(tag_rows)}", flush=True)

    # ---- write tag_counts.csv ----
    with open(out / "tag_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "n_rows", "n_speakers", "mean_duration_s"])
        for t, n in tag_rows.most_common():
            durs = tag_durations[t]
            mean_dur = (sum(durs) / len(durs)) if durs else 0.0
            w.writerow([t, n, len(tag_speakers[t]), f"{mean_dur:.3f}"])

    # ---- tag_cooccurrence.csv ----
    with open(out / "tag_cooccurrence.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag_a", "tag_b", "n_rows"])
        for (a, b), n in sorted(cooc.items(), key=lambda x: -x[1]):
            w.writerow([a, b, n])

    # ---- tag_per_speaker.csv ----
    with open(out / "tag_per_speaker.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speaker_id", "tag", "n_rows"])
        for (spk, t), n in sorted(per_speaker.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
            w.writerow([spk, t, n])

    # ---- neutral_rows_per_speaker.csv ----
    with open(out / "neutral_rows_per_speaker.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speaker_id", "n_neutral_rows"])
        for spk, n in neutral_rows_per_speaker.most_common():
            w.writerow([spk, n])

    # ---- tags_vs_tags_field.json ----
    # For each tag seen in the tags column, what fraction of the time is that same string inline?
    tagfield_agreement = {
        t: {
            "n_in_tags_col": tagcol_seen[t],
            "n_also_inline": inline_seen_for_tagcol_key[t],
            "recall_inline_given_tagcol": (inline_seen_for_tagcol_key[t] / tagcol_seen[t]) if tagcol_seen[t] else 0.0,
        }
        for t in tagcol_seen
    }
    with open(out / "tags_vs_tags_field.json", "w") as f:
        json.dump(tagfield_agreement, f, indent=2)

    # ---- tokenization report ----
    print("[analyze_tags] downloading tokenizer.json ...", flush=True)
    tok_path = hf_hub_download(repo_id="ResembleAI/chatterbox", filename="tokenizer.json")
    tok = EnTokenizer(tok_path)
    unk_id = tok.tokenizer.get_vocab().get(UNK)
    top_tags = [t for t, _ in tag_rows.most_common(args.top_k_for_tokenization)]
    tok_report = {}
    for t in top_tags:
        wrapped = f"[{t}]"
        ids = tok.encode(wrapped)
        pieces = [tok.tokenizer.id_to_token(i) for i in ids]
        stable = (unk_id not in ids) if unk_id is not None else True
        tok_report[t] = {
            "n_tokens": len(ids),
            "token_ids": ids,
            "pieces": pieces,
            "stable": stable,
        }
    with open(out / "tag_tokenization.json", "w") as f:
        json.dump(tok_report, f, indent=2)

    # ---- row_index.jsonl for downstream split builder ----
    # Write once, reuse in build_splits.py to avoid rescanning parquet.
    with open(out / "row_index.jsonl", "w") as f:
        for shard_idx, row_idx, spk, dur, inline, tc in row_index:
            f.write(json.dumps({
                "shard": shard_idx, "row": row_idx, "speaker": spk,
                "duration": dur, "inline_tags": inline, "tags_col": tc,
            }) + "\n")

    # ---- top-40 stdout summary ----
    print()
    print("=" * 70)
    print(f"TOP 40 INLINE-BRACKET TAGS  (total rows={total_rows})")
    print("=" * 70)
    print(f"{'tag':<36} {'rows':>7} {'spks':>6} {'dur/s':>7} {'n_tok':>6} {'stab':>5} {'pieces'}")
    for t, n in tag_rows.most_common(40):
        durs = tag_durations[t]
        md = (sum(durs) / len(durs)) if durs else 0.0
        info = tok_report.get(t, {"n_tokens": -1, "pieces": [], "stable": False})
        print(f"{t:<36} {n:>7} {len(tag_speakers[t]):>6} {md:>7.2f} "
              f"{info['n_tokens']:>6} {str(info['stable']):>5} {info['pieces']}")
    print()
    print(f"rows without any inline tag and empty tags col: "
          f"{sum(neutral_rows_per_speaker.values())} "
          f"(across {len(neutral_rows_per_speaker)} speakers)")
    print(f"artifacts written to {out}/")


if __name__ == "__main__":
    main()
