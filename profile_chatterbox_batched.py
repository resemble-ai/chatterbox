# profile_chatterbox_batched.py
#
# Profiles Chatterbox TTS batched generation using:
#   1. AuraProfiler  — Python (pyinstrument) + Torch (legacy autograd) profiling
#   2. hunt_syncs    — GPU/CPU synchronization detection
#   3. Manual timers — wall-clock timing per phase
#
# Output goes to: ./profiling_output/
#
# Usage:
#   python profile_chatterbox_batched.py
#   python profile_chatterbox_batched.py --prompt path/to/voice.wav
#   python profile_chatterbox_batched.py --no-python     (disable pyinstrument)
#   python profile_chatterbox_batched.py --no-torch      (disable torch trace)
#   python profile_chatterbox_batched.py --no-sync       (disable sync hunting)
#   python profile_chatterbox_batched.py --batch-size 4
#   python profile_chatterbox_batched.py --variations 3
#   python profile_chatterbox_batched.py --compile        (enable torch.compile)
#   python profile_chatterbox_batched.py --bf16           (enable bf16, default on CUDA)

import argparse
import os
import sys
import time
import logging

import torch
import torchaudio as ta

# ── Ensure `tools/` is importable from the script's directory ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from tools.profiling_wrapper import AuraProfiler
from tools.sync_hunter import hunt_syncs

from chatterbox.tts import ChatterboxTTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chatterbox_profiler")


# ─────────────────────────────────────────────────────────
# Test sentences for batching
# ─────────────────────────────────────────────────────────
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Ezreal and Jinx teamed up with Ahri to take down the enemy Nexus.",
    "In a quiet village nestled between mountains, a baker discovers an ancient recipe.",
    "Scientists have announced a breakthrough in quantum computing this morning.",
    "She walked through the garden, admiring the blooming roses and singing birds.",
    "The stock market rallied today after the Federal Reserve held interest rates steady.",
    "Under a canopy of stars, the explorers set up camp beside a crystalline lake.",
    "Breaking news, a rare solar eclipse will be visible across North America next week.",
]

WARMUP_TEXT = "This is a short warmup sentence for the model."


def parse_args():
    p = argparse.ArgumentParser(description="Profile Chatterbox TTS (batched)")
    p.add_argument("--prompt", type=str, default=None,
                   help="Path to audio prompt .wav for voice cloning. Uses default voice if omitted.")
    p.add_argument("--batch-size", type=int, default=2,
                   help="Number of texts per batch (default: 2)")
    p.add_argument("--variations", type=int, default=1,
                   help="num_return_sequences per text (default: 1)")
    p.add_argument("--output-dir", type=str, default="profiling_output",
                   help="Directory for profiling results (default: profiling_output)")
    p.add_argument("--compile", action="store_true",
                   help="Enable torch.compile for T3/S3Gen")
    p.add_argument("--bf16", action="store_true", default=True,
                   help="Use bf16 on CUDA (default: True)")
    p.add_argument("--no-bf16", dest="bf16", action="store_false",
                   help="Force fp32")
    p.add_argument("--no-python", action="store_true",
                   help="Disable pyinstrument Python profiling")
    p.add_argument("--no-torch", action="store_true",
                   help="Disable Torch autograd profiling")
    p.add_argument("--no-sync", action="store_true",
                   help="Disable GPU/CPU sync hunting")
    p.add_argument("--warmup-runs", type=int, default=1,
                   help="Number of warmup generation passes (default: 1)")
    p.add_argument("--save-audio", action="store_true",
                   help="Save generated .wav files alongside profiling results")
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print_gpu_info(device):
    if device != "cuda":
        logger.info(f"Device: {device}")
        return
    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info(f"Device: {device} — {gpu_name} ({vram_total:.1f} GB VRAM)")


def write_timing_report(timings: dict, output_dir: str, args):
    """Write a human-readable wall-clock timing report."""
    report_path = os.path.join(output_dir, "timing_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  CHATTERBOX TTS — BATCHED PROFILING TIMING REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Device:             {timings.get('device', '?')}\n")
        f.write(f"Dtype:              {timings.get('dtype', '?')}\n")
        f.write(f"torch.compile:      {args.compile}\n")
        f.write(f"Batch size:         {args.batch_size}\n")
        f.write(f"Variations:         {args.variations}\n")
        f.write(f"Audio prompt:       {args.prompt or '(default voice)'}\n\n")

        f.write("-" * 60 + "\n")
        f.write(f"{'Phase':<35} {'Time (s)':>10}\n")
        f.write("-" * 60 + "\n")
        for phase, dur in timings.items():
            if phase in ("device", "dtype"):
                continue
            f.write(f"{phase:<35} {dur:>10.3f}\n")
        f.write("-" * 60 + "\n")

        # Derived metrics
        gen_time = timings.get("generate_batched", 0)
        n_texts = args.batch_size
        n_total = n_texts * args.variations
        if gen_time > 0 and n_total > 0:
            f.write(f"\n{'Avg time per output':.<35} {gen_time / n_total:>10.3f} s\n")
            f.write(f"{'Throughput':.<35} {n_total / gen_time:>10.2f} outputs/s\n")

        # VRAM snapshot (CUDA only)
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
            reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)
            f.write(f"\n{'Peak VRAM allocated':.<35} {peak_mb:>10.1f} MB\n")
            f.write(f"{'Peak VRAM reserved':.<35} {reserved_mb:>10.1f} MB\n")

    logger.info(f"Timing report saved to {report_path}")


def main():
    args = parse_args()
    device = get_device()
    print_gpu_info(device)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    timings = {"device": device}

    # ── 1. Load Model ──
    logger.info("Loading Chatterbox TTS model...")
    t0 = time.perf_counter()

    use_bf16 = args.bf16 and device == "cuda"
    model = ChatterboxTTS.from_pretrained(
        device=device,
        use_bf16=use_bf16,
        compile_model=args.compile,
    )
    timings["dtype"] = str(model.dtype)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings["model_load"] = time.perf_counter() - t0
    logger.info(f"Model loaded in {timings['model_load']:.2f}s  (dtype={model.dtype})")

    # ── 2. Prepare Conditionals (voice prompt) ──
    if args.prompt:
        logger.info(f"Preparing conditionals from: {args.prompt}")
        t0 = time.perf_counter()
        model.prepare_conditionals(args.prompt)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings["prepare_conditionals"] = time.perf_counter() - t0
        logger.info(f"Conditionals ready in {timings['prepare_conditionals']:.2f}s")

    # ── 3. Warmup (outside profiling) ──
    if args.warmup_runs > 0:
        logger.info(f"Running {args.warmup_runs} warmup pass(es)...")
        t0 = time.perf_counter()
        for _ in range(args.warmup_runs):
            if args.prompt:
                _ = model.generate(WARMUP_TEXT, audio_prompt_path=args.prompt)
            else:
                _ = model.generate(WARMUP_TEXT)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings["warmup"] = time.perf_counter() - t0
        logger.info(f"Warmup done in {timings['warmup']:.2f}s")

    # Reset peak VRAM tracking after warmup
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ── 4. Setup Profilers ──
    profiler = AuraProfiler(
        log_dir=output_dir,
        actor_name="chatterbox_batch",
        enabled=True,
        enable_python=not args.no_python,
        enable_torch=not args.no_torch,
        # repeat=0 → legacy autograd profiler (no Kineto schedule)
        schedule_config={"repeat": 0} if not args.no_torch else None,
    )

    sync_log_path = os.path.join(output_dir, "sync_report.txt")

    # ── 5. Profiled Batched Generation ──
    batch_texts = SAMPLE_TEXTS[:args.batch_size]
    logger.info(f"Generating batch of {len(batch_texts)} texts × {args.variations} variation(s)...")

    gen_kwargs = dict(
        text=batch_texts,
        num_return_sequences=args.variations,
    )
    if args.prompt:
        gen_kwargs["audio_prompt_path"] = args.prompt

    with hunt_syncs(enabled=not args.no_sync, log_file=sync_log_path):
        profiler.start()
        t0 = time.perf_counter()

        wavs = model.generate(**gen_kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings["generate_batched"] = time.perf_counter() - t0

        profiler.stop_and_save(tag="batch_generate")

    logger.info(f"Batched generation: {timings['generate_batched']:.2f}s")

    # ── 6. Save Audio (optional) ──
    if args.save_audio:
        audio_dir = os.path.join(output_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        if args.variations > 1:
            for i, group in enumerate(wavs):
                for j, wav in enumerate(group):
                    fpath = os.path.join(audio_dir, f"batch_{i+1}_var_{j+1}.wav")
                    ta.save(fpath, wav, model.sr)
        else:
            if isinstance(wavs, list):
                for i, wav in enumerate(wavs):
                    fpath = os.path.join(audio_dir, f"batch_{i+1}.wav")
                    ta.save(fpath, wav, model.sr)
            else:
                ta.save(os.path.join(audio_dir, "output.wav"), wavs, model.sr)
        logger.info(f"Audio saved to {audio_dir}/")

    # ── 7. Write Timing Report ──
    write_timing_report(timings, output_dir, args)

    # ── 8. Summary ──
    logger.info("")
    logger.info("=" * 50)
    logger.info("  PROFILING COMPLETE — Output files:")
    logger.info("=" * 50)
    for f in sorted(os.listdir(output_dir)):
        full = os.path.join(output_dir, f)
        if os.path.isfile(full):
            size_kb = os.path.getsize(full) / 1024
            logger.info(f"  {f:<45} ({size_kb:.1f} KB)")
    # Also check the profiler subdirectory
    prof_sub = os.path.join(output_dir, "profiling", "chatterbox_batch")
    if os.path.isdir(prof_sub):
        for f in sorted(os.listdir(prof_sub)):
            full = os.path.join(prof_sub, f)
            if os.path.isfile(full):
                size_kb = os.path.getsize(full) / 1024
                logger.info(f"  profiling/chatterbox_batch/{f:<25} ({size_kb:.1f} KB)")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()