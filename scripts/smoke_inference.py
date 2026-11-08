"""Generate a reproducible audio fixture and record basic signal health."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import time

import numpy as np
import soundfile as sf
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["english", "multilingual", "turbo", "nano"],
                        default="english")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-dir", type=Path)
    source.add_argument("--repo-id")
    parser.add_argument("--revision", help="Immutable mirror/upstream commit SHA")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--text", default="Hello. This is a test of my maintained Chatterbox library.")
    parser.add_argument("--language", default="en")
    parser.add_argument("--t3-model", default="v3")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke.wav"))
    args = parser.parse_args()
    if args.repo_id and not args.revision:
        parser.error("--repo-id requires --revision so the result has model provenance")
    from chatterbox import ChatterboxMultilingualTTS, ChatterboxTTS, ChatterboxTurboTTS

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    classes = {"english": ChatterboxTTS, "multilingual": ChatterboxMultilingualTTS,
               "turbo": ChatterboxTurboTTS, "nano": ChatterboxTurboTTS}
    kwargs = {"device": args.device}
    if args.variant in {"turbo", "nano"}:
        kwargs["nano"] = args.variant == "nano"
    elif args.variant == "multilingual":
        kwargs["t3_model"] = args.t3_model
    start = time.monotonic()
    cls = classes[args.variant]
    if args.model_dir:
        model = cls.from_local(args.model_dir, **kwargs)
    else:
        model = cls.from_pretrained(repo_id=args.repo_id, revision=args.revision, **kwargs)
    load_seconds = time.monotonic() - start
    generate_kwargs = {}
    if args.reference:
        generate_kwargs["audio_prompt_path"] = str(args.reference)
    if args.variant == "multilingual":
        generate_kwargs["language_id"] = args.language
    start = time.monotonic()
    wav = model.generate(args.text, **generate_kwargs).detach().cpu().numpy().reshape(-1)
    generation_seconds = time.monotonic() - start
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    if wav.size == 0 or not np.isfinite(wav).all() or not np.isfinite(rms) or rms < 1e-6:
        raise RuntimeError("Generated audio is empty, nonfinite, or silent")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, wav, model.sr, subtype="PCM_16")
    provenance = {}
    manifest_hash = None
    if args.model_dir:
        manifest_path = args.model_dir / "_chatterbox_backup/manifest.json"
        if manifest_path.is_file():
            provenance = json.loads(manifest_path.read_text())["upstream"]
            manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    report = {
        "variant": args.variant, "device": str(model.device), "seed": args.seed,
        "text": args.text, "language": args.language, "t3_model": args.t3_model,
        "repo_id": args.repo_id or provenance.get("repo_id"),
        "revision": args.revision or provenance.get("revision"),
        "archive_manifest_sha256": manifest_hash,
        "model_dir": str(args.model_dir) if args.model_dir else None,
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest()
            if args.reference else None,
        "sample_rate": model.sr, "samples": wav.size, "duration_seconds": wav.size / model.sr,
        "rms": rms, "peak": float(np.abs(wav).max()),
        "load_seconds": load_seconds, "generation_seconds": generation_seconds,
        "audio_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "packages": {name: version(name) for name in
                     ["chatterbox-tts", "torch", "torchaudio", "transformers", "resemble-perth"]},
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
