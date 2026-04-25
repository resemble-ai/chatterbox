"""Extract 5 audio samples from the same speaker to verify speaker identification.

Usage examples
--------------
# List all voice_ids with their locale and row counts:
.venv/bin/python analysis/verify_speakers.py --list

# Extract 5 wavs from the default first speaker found:
.venv/bin/python analysis/verify_speakers.py --out_dir /tmp/speaker_check/

# Extract from a specific voice_id:
.venv/bin/python analysis/verify_speakers.py --voice_id 693c79d60bee6 --out_dir /tmp/speaker_check/

# Pick by locale (gets the first speaker for that locale):
.venv/bin/python analysis/verify_speakers.py --locale english_british --out_dir /tmp/speaker_check/

# Pick by locale + gender:
.venv/bin/python analysis/verify_speakers.py --locale english_australian --gender female --out_dir /tmp/speaker_check/
"""
from __future__ import annotations
import argparse, io, json, re
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
import torchaudio
import torch

SNAPSHOT = (
    "/home/sruthi/.cache/huggingface/hub/"
    "datasets--InternalCan--stage1-processed-with-audio-aligned/"
    "snapshots/5339999e2931ec74bbe2c845db6fc48383e1a549"
)
ROW_INDEX = "/home/sruthi/chatterbox/analysis/row_index.jsonl"


def load_row_index():
    rows = []
    with open(ROW_INDEX) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def get_shard_path(shard_idx: int) -> str:
    import glob
    shards = sorted(glob.glob(f"{SNAPSHOT}/data/train-*.parquet"))
    return shards[shard_idx]


def read_audio_bytes(shard_idx: int, row_idx: int) -> tuple[bytes, str]:
    """Return (wav_bytes, script_content) for a given shard/row."""
    shard_path = get_shard_path(shard_idx)
    pf = pq.ParquetFile(shard_path)
    # find which row group contains row_idx
    rows_seen = 0
    for rg in range(pf.num_row_groups):
        rg_size = pf.metadata.row_group(rg).num_rows
        if rows_seen + rg_size > row_idx:
            local_idx = row_idx - rows_seen
            tbl = pf.read_row_group(rg, columns=["preprocessed_audio", "script_content"]).slice(local_idx, 1).to_pydict()
            audio_struct = tbl["preprocessed_audio"][0]
            wav_bytes = audio_struct["bytes"] if audio_struct else None
            script = tbl["script_content"][0] or ""
            return wav_bytes, script
        rows_seen += rg_size
    raise IndexError(f"row_idx {row_idx} out of range for shard {shard_idx}")


def bytes_to_wav(raw: bytes) -> tuple[torch.Tensor, int]:
    """Decode raw bytes -> (waveform_tensor, sample_rate)."""
    data, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # mono
    wav = torch.from_numpy(data).unsqueeze(0)  # (1, T)
    return wav, sr


def main():
    ap = argparse.ArgumentParser(description="Verify speaker ID extraction by sampling audio from one speaker.")
    ap.add_argument("--list", action="store_true", help="List all voice_ids, locales, and row counts then exit")
    ap.add_argument("--voice_id", type=str, default=None, help="Specific voice_id to sample (13-char hex prefix)")
    ap.add_argument("--locale", type=str, default=None, help="Filter by locale (e.g. english_british)")
    ap.add_argument("--gender", type=str, default=None, help="Filter by gender (male/female/neutral)")
    ap.add_argument("--n", type=int, default=5, help="Number of samples to extract (default 5)")
    ap.add_argument("--out_dir", type=str, default="/tmp/speaker_check/")
    ap.add_argument("--row_index", type=str, default=ROW_INDEX)
    args = ap.parse_args()

    print("Loading row index ...")
    rows = load_row_index()

    # build speaker registry
    by_voice: dict[str, list[dict]] = defaultdict(list)  # voice_id -> list of row metadata
    for idx, r in enumerate(rows):
        r["_row_index"] = idx  # stash flat index
        by_voice[r["voice_id"]].append(r)

    # ---- --list mode ----
    if args.list:
        print(f"\n{'voice_id':<20} {'locale':<55} {'rows':>6} {'gender_dist'}")
        for vid in sorted(by_voice.keys()):
            items = by_voice[vid]
            locale = items[0]["locale"]
            genders = defaultdict(int)
            for r in items:
                genders[r["gender"]] += 1
            print(f"{vid:<20} {locale:<55} {len(items):>6}  {dict(genders)}")
        print(f"\nTotal: {len(by_voice)} speakers")
        return

    # ---- select speaker ----
    if args.voice_id:
        if args.voice_id not in by_voice:
            # try prefix match
            matches = [v for v in by_voice if v.startswith(args.voice_id)]
            if not matches:
                print(f"No voice_id matching '{args.voice_id}' found.")
                print("Run with --list to see available speakers.")
                return
            chosen_vid = matches[0]
        else:
            chosen_vid = args.voice_id
    else:
        # filter by locale / gender
        candidates = list(by_voice.keys())
        if args.locale:
            candidates = [v for v in candidates if by_voice[v][0]["locale"] == args.locale]
        if args.gender:
            candidates = [v for v in candidates
                          if any(r["gender"] == args.gender for r in by_voice[v])]
        if not candidates:
            print(f"No speakers found matching locale={args.locale} gender={args.gender}")
            return
        chosen_vid = sorted(candidates)[0]

    speaker_rows = by_voice[chosen_vid]
    locale = speaker_rows[0]["locale"]
    print(f"\nChosen speaker: {chosen_vid}  locale={locale}  total_rows={len(speaker_rows)}")
    print(f"Sampling {args.n} rows ...\n")

    import random; random.seed(0)
    sample = random.sample(speaker_rows, min(args.n, len(speaker_rows)))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(sample):
        shard_idx = r["shard"]
        row_idx   = r["row"]
        try:
            wav_bytes, script = read_audio_bytes(shard_idx, row_idx)
        except Exception as e:
            print(f"  [{i}] ERROR reading shard={shard_idx} row={row_idx}: {e}")
            continue

        if not wav_bytes:
            print(f"  [{i}] SKIP — no audio bytes at shard={shard_idx} row={row_idx}")
            continue

        wav, sr = bytes_to_wav(wav_bytes)
        out_path = out / f"{chosen_vid}_{i:02d}.wav"
        torchaudio.save(str(out_path), wav, sr)

        tags = r.get("inline_tags", [])
        script_preview = script[:140].replace("\n", " ")
        print(f"  [{i}] {out_path.name}  sr={sr}  dur={wav.shape[-1]/sr:.1f}s  tags={tags}")
        print(f"       script: {script_preview}")
        print()

    print(f"Wrote {len(sample)} wav files to {out}/")
    print(f"\nTo pick a different speaker, run:")
    print(f"  .venv/bin/python analysis/verify_speakers.py --list")
    print(f"  .venv/bin/python analysis/verify_speakers.py --voice_id <voice_id> --out_dir {out}/")
    print(f"\nOther examples:")
    print(f"  # British English speaker:")
    print(f"  .venv/bin/python analysis/verify_speakers.py --locale english_british --out_dir {out}/")
    print(f"  # Female Australian speaker:")
    print(f"  .venv/bin/python analysis/verify_speakers.py --locale english_australian --gender female --out_dir {out}/")
    print(f"  # Indian English:")
    print(f"  .venv/bin/python analysis/verify_speakers.py --locale english_indian-india --out_dir {out}/")


if __name__ == "__main__":
    main()
