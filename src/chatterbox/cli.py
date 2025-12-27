#!/usr/bin/env python3
"""
Chatterbox MLX Command Line Interface
=====================================

Simple TTS generation from the command line.

Examples:
    Generate English speech (auto-generated filename):
    chatterbox "Artificial intelligence has made remarkable strides in recent years, particularly in the field of natural language processing."

    Generate Spanish speech:
    chatterbox "La inteligencia artificial ha logrado avances notables en los últimos años." --lang es

    Use the --voice flag to provide a reference audio file for voice cloning:
    chatterbox "Artificial intelligence has made remarkable strides in recent years, particularly in the field of natural language processing." --voice speaker.wav

    Run a quick multilingual benchmark:
    chatterbox --benchmark
"""

import argparse
import sys
import time
from pathlib import Path


# Benchmark test texts for different languages
BENCHMARK_TEXTS = {
    "en": (
        "English",
        "Hello, this is a test of multilingual speech synthesis. The technology can generate natural sounding voices in many different languages.",
    ),
    "es": (
        "Spanish",
        "Hola, esta es una prueba de síntesis de voz multilingüe. La tecnología puede generar voces de sonido natural en muchos idiomas diferentes.",
    ),
    "fr": (
        "French",
        "Bonjour, ceci est un test de synthèse vocale multilingue. La technologie peut générer des voix naturelles dans de nombreuses langues différentes.",
    ),
    "de": (
        "German",
        "Hallo, dies ist ein Test der mehrsprachigen Sprachsynthese. Die Technologie kann natürlich klingende Stimmen in vielen verschiedenen Sprachen erzeugen.",
    ),
    "ja": (
        "Japanese",
        "こんにちは、これは多言語音声合成のテストです。この技術は、さまざまな言語で自然な音声を生成できます。",
    ),
    "zh": (
        "Chinese",
        "你好，这是多语言语音合成的测试。该技术可以生成许多不同语言的自然语音。",
    ),
}


def run_benchmark(
    languages=None, backend="hybrid-mlx", save_audio=True, output_dir="benchmark_output"
):
    """Run a quick multilingual benchmark."""
    import torchaudio as ta

    if languages is None:
        languages = ["en", "es", "fr", "de", "ja", "zh"]

    print("=" * 60)
    print("🎯 CHATTERBOX MLX BENCHMARK")
    print("=" * 60)
    print(f"   Backend: {backend}")
    print(f"   Languages: {', '.join(languages)}")
    print("=" * 60)
    print()

    # Load model
    print("⏳ Loading model...")
    load_start = time.time()

    if backend in ("hybrid-mlx", "mlx"):
        from chatterbox.mtl_tts_mlx import ChatterboxMultilingualTTSMLX

        model = ChatterboxMultilingualTTSMLX.from_pretrained()
    else:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        model = ChatterboxMultilingualTTS.from_pretrained(device="mps")

    load_time = time.time() - load_start
    print(f"   Model loaded in {load_time:.1f}s")
    print()

    # Create output dir if saving
    if save_audio:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = []
    total_gen_time = 0
    total_duration = 0

    print("🔊 Running benchmark...")
    print("-" * 60)

    for lang in languages:
        if lang not in BENCHMARK_TEXTS:
            print(f"   ⚠️  Skipping unknown language: {lang}")
            continue

        lang_name, text = BENCHMARK_TEXTS[lang]
        print(f'   [{lang}] {lang_name}: "{text[:40]}..."')

        gen_start = time.time()
        wav = model.generate(text, language_id=lang)
        gen_time = time.time() - gen_start

        duration = wav.shape[-1] / model.sr
        rtf = duration / gen_time

        total_gen_time += gen_time
        total_duration += duration

        results.append(
            {
                "lang": lang,
                "name": lang_name,
                "gen_time": gen_time,
                "duration": duration,
                "rtf": rtf,
            }
        )

        print(
            f"       → {gen_time:.2f}s generation, {duration:.1f}s audio, RTF: {rtf:.2f}x"
        )

        if save_audio:
            output_path = Path(output_dir) / f"benchmark_{lang}.wav"
            ta.save(str(output_path), wav, model.sr)

    print("-" * 60)
    print()

    # Summary
    avg_rtf = total_duration / total_gen_time if total_gen_time > 0 else 0

    print("=" * 60)
    print("📊 BENCHMARK RESULTS")
    print("=" * 60)
    print(f"{'Language':<12} {'Gen Time':>10} {'Duration':>10} {'RTF':>8}")
    print("-" * 60)

    for r in results:
        print(
            f"{r['name']:<12} {r['gen_time']:>9.2f}s {r['duration']:>9.1f}s {r['rtf']:>7.2f}x"
        )

    print("-" * 60)
    print(
        f"{'TOTAL':<12} {total_gen_time:>9.2f}s {total_duration:>9.1f}s {avg_rtf:>7.2f}x"
    )
    print("=" * 60)

    if save_audio:
        print(f"\n📁 Audio files saved to: {output_dir}/")

    return 0


def get_default_device():
    """Auto-detect the best available device."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def generate_output_filename(text: str, lang: str) -> str:
    """Generate a sensible output filename."""
    # Use first few words of text, sanitized
    words = text.split()[:4]
    slug = "_".join(words)
    # Remove non-alphanumeric characters
    slug = "".join(c if c.isalnum() or c == "_" else "" for c in slug)
    slug = slug[:30]  # Limit length
    if not slug:
        slug = f"output_{int(time.time())}"
    return f"{slug}_{lang}.wav"


def main():
    parser = argparse.ArgumentParser(
        prog="chatterbox",
        description="Generate speech from text using Chatterbox MLX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  chatterbox "Hello, world!"
  chatterbox "Hola, cómo estás?" --lang es
  chatterbox "Bonjour!" --lang fr -o french.wav
  chatterbox "Hello" --voice reference.wav --exaggeration 0.7
  chatterbox --benchmark
  chatterbox --benchmark --languages en es ja --save-audio

Supported Languages:
  en (English), es (Spanish), fr (French), de (German), it (Italian),
  pt (Portuguese), ru (Russian), ja (Japanese), zh (Chinese), ko (Korean),
  ar (Arabic), hi (Hindi), and more...
        """,
    )

    parser.add_argument("text", nargs="?", help="Text to convert to speech")
    parser.add_argument(
        "--benchmark", action="store_true", help="Run multilingual benchmark"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "es", "fr", "de", "ja", "zh"],
        help="Languages to benchmark (default: en es fr de ja zh)",
    )
    parser.add_argument(
        "--no-save-audio",
        action="store_true",
        help="Don't save benchmark audio files (default: save to benchmark_output/)",
    )
    parser.add_argument(
        "-o", "--output", help="Output WAV file path (auto-generated if not specified)"
    )
    parser.add_argument(
        "-l",
        "--lang",
        "--language",
        default="en",
        help="Language code (default: en). Examples: es, fr, de, ja, zh",
    )
    parser.add_argument("-v", "--voice", help="Reference audio file for voice cloning")
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="Emotion intensity 0.0-1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=0.5,
        help="Classifier-free guidance weight (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device: mps, cuda, cpu (auto-detected if not specified)",
    )
    parser.add_argument(
        "--backend",
        choices=["hybrid-mlx", "mlx", "pytorch"],
        default="hybrid-mlx",
        help="Backend to use (default: hybrid-mlx)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress progress messages"
    )

    args = parser.parse_args()

    # Handle benchmark mode
    if args.benchmark:
        return run_benchmark(
            languages=args.languages,
            backend=args.backend,
            save_audio=not args.no_save_audio,
        )

    # Regular TTS mode - text is required
    if not args.text:
        parser.error("text is required (or use --benchmark)")

    # Auto-detect device if not specified
    device = args.device or get_default_device()

    # Generate output filename if not specified
    output_path = args.output or generate_output_filename(args.text, args.lang)

    if not args.quiet:
        print("🎤 Chatterbox MLX")
        print(f"   Text: \"{args.text[:50]}{'...' if len(args.text) > 50 else ''}\"")
        print(f"   Language: {args.lang}")
        print(f"   Backend: {args.backend}")
        print(f"   Output: {output_path}")
        print()

    # Validate voice file exists before loading model
    if args.voice and not Path(args.voice).exists():
        print(f"❌ Error: Voice file not found: '{args.voice}'", file=sys.stderr)
        print(
            "   Please provide a valid path to a WAV file for voice cloning.",
            file=sys.stderr,
        )
        return 1

    try:
        import torchaudio as ta

        # Load the appropriate model
        if not args.quiet:
            print("⏳ Loading model...")

        load_start = time.time()

        if args.backend in ("hybrid-mlx", "mlx"):
            # MLX backends: Always use multilingual model (supports all languages including English)
            # MLX models don't take a device parameter - they auto-detect
            from chatterbox.mtl_tts_mlx import ChatterboxMultilingualTTSMLX

            model = ChatterboxMultilingualTTSMLX.from_pretrained()
        else:  # pytorch
            if args.lang != "en":
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS

                model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            else:
                from chatterbox.tts import ChatterboxTTS

                model = ChatterboxTTS.from_pretrained(device=device)

        load_time = time.time() - load_start

        if not args.quiet:
            print(f"   Model loaded in {load_time:.1f}s")
            print("🔊 Generating speech...")

        # Generate audio
        gen_kwargs = {
            "exaggeration": args.exaggeration,
            "cfg_weight": args.cfg,
        }

        if args.voice:
            gen_kwargs["audio_prompt_path"] = args.voice

        # MLX multilingual model always needs language_id
        # PyTorch multilingual model needs language_id for non-English
        if args.backend in ("hybrid-mlx", "mlx"):
            gen_kwargs["language_id"] = args.lang
        elif args.lang != "en":
            gen_kwargs["language_id"] = args.lang

        # Only time the actual generation
        gen_start = time.time()
        wav = model.generate(args.text, **gen_kwargs)
        gen_time = time.time() - gen_start

        # Save audio
        ta.save(output_path, wav, model.sr)

        if not args.quiet:
            duration = wav.shape[-1] / model.sr
            rtf = duration / gen_time
            print(f"✅ Saved to {output_path}")
            print(
                f"   Duration: {duration:.1f}s | Generated in {gen_time:.1f}s | RTF: {rtf:.2f}x"
            )
        else:
            print(output_path)

        return 0

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
