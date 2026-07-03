"""
Chatterbox CLI — Command-line interface for TTS, Turbo TTS, Multilingual TTS, and Voice Conversion.

Usage:
    chatterbox tts --text "Hello world" --ref-audio speaker.wav --output out.wav
    chatterbox turbo --text "Hello world" --ref-audio speaker.wav --output out.wav
    chatterbox multilingual --text "Bonjour" --lang fr --ref-audio speaker.wav --output out.wav
    chatterbox vc --source-audio input.wav --target-voice speaker.wav --output out.wav
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import torchaudio as ta

from . import __version__


def _auto_device() -> str:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def _validate_file(path: str, label: str):
    """Check that a file exists before loading the model."""
    if path is not None and not os.path.isfile(path):
        print(f"Error: {label} file not found: {path}", file=sys.stderr)
        sys.exit(1)


def _add_common_args(parser: argparse.ArgumentParser):
    """Add arguments common to all TTS subcommands."""
    parser.add_argument(
        "--text", "-t",
        type=str,
        default=None,
        help="Text to synthesize (required for generation).",
    )
    parser.add_argument(
        "--ref-audio", "-r",
        type=str,
        default=None,
        help="Path to reference audio file for voice cloning. Uses built-in voice if not specified.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.wav",
        help="Output audio file path (default: output.wav).",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default=None,
        help="Device to use: cuda, cpu, or mps (default: auto-detect).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8).",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Repetition penalty (default: 1.2).",
    )


def _handle_tts(args):
    """Handle the 'tts' subcommand (English TTS)."""
    from .tts import ChatterboxTTS

    if not args.text:
        print("Error: --text is required.", file=sys.stderr)
        sys.exit(1)
    _validate_file(args.ref_audio, "--ref-audio")

    device = args.device or _auto_device()
    if args.seed is not None:
        _set_seed(args.seed)

    print(f"Loading ChatterboxTTS on {device}...")
    model = ChatterboxTTS.from_pretrained(device)

    print(f"Generating speech...")
    wav = model.generate(
        text=args.text,
        audio_prompt_path=args.ref_audio,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
    )

    ta.save(args.output, wav, model.sr)
    print(f"Saved to {args.output}")


def _handle_turbo(args):
    """Handle the 'turbo' subcommand (low-latency English TTS)."""
    from .tts_turbo import ChatterboxTurboTTS

    if not args.text:
        print("Error: --text is required.", file=sys.stderr)
        sys.exit(1)
    if args.ref_audio is None:
        print("Error: --ref-audio is required for the Turbo model.", file=sys.stderr)
        sys.exit(1)
    _validate_file(args.ref_audio, "--ref-audio")

    device = args.device or _auto_device()
    if args.seed is not None:
        _set_seed(args.seed)

    print(f"Loading ChatterboxTurboTTS on {device}...")
    model = ChatterboxTurboTTS.from_pretrained(device)

    print(f"Generating speech...")
    wav = model.generate(
        text=args.text,
        audio_prompt_path=args.ref_audio,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    ta.save(args.output, wav, model.sr)
    print(f"Saved to {args.output}")


def _handle_multilingual(args):
    """Handle the 'multilingual' subcommand (Multilingual TTS)."""
    from .mtl_tts import ChatterboxMultilingualTTS

    _validate_file(args.ref_audio, "--ref-audio")

    device = args.device or _auto_device()
    if args.seed is not None:
        _set_seed(args.seed)

    print(f"Loading ChatterboxMultilingualTTS on {device}...")
    model = ChatterboxMultilingualTTS.from_pretrained(device, t3_model=args.t3_model)

    print(f"Generating speech (language: {args.lang})...")
    wav = model.generate(
        text=args.text,
        language_id=args.lang,
        audio_prompt_path=args.ref_audio,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
    )

    ta.save(args.output, wav, model.sr)
    print(f"Saved to {args.output}")


def _handle_vc(args):
    """Handle the 'vc' subcommand (Voice Conversion)."""
    from .vc import ChatterboxVC

    _validate_file(args.source_audio, "--source-audio")
    _validate_file(args.target_voice, "--target-voice")

    device = args.device or _auto_device()

    print(f"Loading ChatterboxVC on {device}...")
    model = ChatterboxVC.from_pretrained(device)

    print(f"Converting voice...")
    wav = model.generate(
        audio=args.source_audio,
        target_voice_path=args.target_voice,
    )

    ta.save(args.output, wav, model.sr)
    print(f"Saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        prog="chatterbox",
        description="Chatterbox TTS — Open-source Text-to-Speech and Voice Conversion by Resemble AI.",
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"chatterbox-tts {__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands:",
    )

    # ── tts ──────────────────────────────────────────────────────────────
    tts_parser = subparsers.add_parser(
        "tts",
        help="English text-to-speech (ChatterboxTTS, 500M).",
    )
    _add_common_args(tts_parser)
    tts_parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="Exaggeration level, 0.5 = neutral (default: 0.5).",
    )
    tts_parser.add_argument(
        "--cfg-weight",
        type=float,
        default=0.5,
        help="Classifier-Free Guidance weight (default: 0.5).",
    )
    tts_parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling (default: 1.0).",
    )
    tts_parser.add_argument(
        "--min-p",
        type=float,
        default=0.05,
        help="Min-p sampling threshold (default: 0.05).",
    )
    tts_parser.set_defaults(func=_handle_tts)

    # ── turbo ────────────────────────────────────────────────────────────
    turbo_parser = subparsers.add_parser(
        "turbo",
        help="Low-latency English TTS with paralinguistic tags (ChatterboxTurboTTS, 350M).",
    )
    _add_common_args(turbo_parser)
    turbo_parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="Top-k sampling (default: 1000).",
    )
    turbo_parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling (default: 0.95).",
    )
    turbo_parser.set_defaults(func=_handle_turbo)

    # ── multilingual ─────────────────────────────────────────────────────
    mtl_parser = subparsers.add_parser(
        "multilingual",
        help="Multilingual TTS supporting 23+ languages (ChatterboxMultilingualTTS, 500M).",
    )
    _add_common_args(mtl_parser)
    mtl_parser.add_argument(
        "--lang", "-l",
        type=str,
        default=None,
        help="Language code (e.g., en, fr, de, zh, ja, ko, hi, ar). Use 'chatterbox multilingual --list-languages' to see all.",
    )
    mtl_parser.add_argument(
        "--t3-model",
        type=str,
        default=None,
        help="Multilingual T3 model variant: v2, v3 (default: v2).",
    )
    mtl_parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="Exaggeration level (default: 0.5).",
    )
    mtl_parser.add_argument(
        "--cfg-weight",
        type=float,
        default=0.5,
        help="Classifier-Free Guidance weight (default: 0.5).",
    )
    mtl_parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling (default: 1.0).",
    )
    mtl_parser.add_argument(
        "--min-p",
        type=float,
        default=0.05,
        help="Min-p sampling threshold (default: 0.05).",
    )
    mtl_parser.add_argument(
        "--list-languages",
        action="store_true",
        help="Print all supported languages and exit.",
    )
    mtl_parser.set_defaults(func=_handle_multilingual_or_list)

    # ── vc ───────────────────────────────────────────────────────────────
    vc_parser = subparsers.add_parser(
        "vc",
        help="Voice conversion (ChatterboxVC).",
    )
    vc_parser.add_argument(
        "--source-audio", "-s",
        type=str,
        required=True,
        help="Path to source audio file to convert.",
    )
    vc_parser.add_argument(
        "--target-voice", "-r",
        type=str,
        required=True,
        help="Path to target voice reference audio file.",
    )
    vc_parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.wav",
        help="Output audio file path (default: output.wav).",
    )
    vc_parser.add_argument(
        "--device", "-d",
        type=str,
        default=None,
        help="Device to use: cuda, cpu, or mps (default: auto-detect).",
    )
    vc_parser.set_defaults(func=_handle_vc)

    # ── parse and dispatch ───────────────────────────────────────────────
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


def _handle_multilingual_or_list(args):
    """Wrapper that handles --list-languages before delegating to the real handler."""
    if getattr(args, "list_languages", False):
        from .mtl_tts import SUPPORTED_LANGUAGES
        print("Supported languages:")
        for code, name in sorted(SUPPORTED_LANGUAGES.items()):
            print(f"  {code}  {name}")
        return

    # --lang and --text are required when not listing languages
    if not args.lang:
        print("Error: --lang is required for multilingual TTS. Use --list-languages to see options.", file=sys.stderr)
        sys.exit(1)
    if not args.text:
        print("Error: --text is required.", file=sys.stderr)
        sys.exit(1)

    _handle_multilingual(args)


if __name__ == "__main__":
    main()
