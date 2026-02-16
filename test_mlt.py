"""
Streaming TTS profiler — breaks down where decode time goes,
then tests parameter variations to find the fastest config.

Key features:
- Phase 1: vocoder breakdown with CUDA sync:
    FLOW (GPU), CROP (CPU), HIFIGAN (GPU), CPU_COPY (CPU), WATERMARK (CPU), OVERHEAD
- Phase 2: parameter sweep:
    For each config: 1 discard + N timed runs, report MEDIAN.
- Saves WAVs: profile_<label>_run<k>.wav
- Captures "attention fallback" warnings and prints a summary.
- Prints implied non-vocoder time estimate: wall_time - vocoder_total

Usage:
    python test_stream_profile.py
"""

import os
import re
import time
import warnings
import logging
import statistics
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
import soundfile as sf

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# ---------------------------
# Settings
# ---------------------------

DEVICE = os.getenv("DEVICE", "cuda")
LANG = os.getenv("LANG", "en")
TEXT = os.getenv(
    "TEXT",
    "The weather today is absolutely beautiful, perfect for a walk in the park."
)

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "400"))
TIMED_RUNS = int(os.getenv("TIMED_RUNS", "3"))       # after discard
DISCARD_RUNS = int(os.getenv("DISCARD_RUNS", "1"))
SAVE_WAV = os.getenv("SAVE_WAV", "1") == "1"

# If you want apples-to-apples with the repo’s README example, set TEXT to their example
# and keep similar streaming params.

# ---------------------------
# Torch perf knobs
# ---------------------------

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

if torch.cuda.is_available() and DEVICE.startswith("cuda"):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# Reduce noise from logging
logging.getLogger().setLevel(logging.WARNING)

# ---------------------------
# Warning capture helper
# ---------------------------

ATTN_FALLBACK_PATTERNS = [
    r"Falling back to the manual attention implementation",
    r"scaled_dot_product_attention.*does not support `output_attentions=True`",
    r"sdpa_kernel\(\) is deprecated",
    r"past_key_values.*deprecated",
]

@dataclass
class RunResult:
    label: str
    chunks: int
    first_chunk_s: Optional[float]
    first_token_s: Optional[float]
    first_decode_s: Optional[float]
    total_gen_s: Optional[float]
    audio_dur_s: float
    rtf: Optional[float]
    wall_s: float
    # extra debug
    attn_fallback_hits: int
    attn_fallback_lines: List[str]
    vocoder_total_s: Optional[float] = None
    implied_non_vocoder_s: Optional[float] = None

def _cuda_sync():
    if torch.cuda.is_available() and torch.cuda.current_device() is not None:
        torch.cuda.synchronize()

class WarningSniffer:
    """Capture warnings + select stderr-like messages into a list."""
    def __init__(self):
        self.lines: List[str] = []
        self.hits = 0

    def reset(self):
        self.lines.clear()
        self.hits = 0

    def observe(self, msg: str):
        if not msg:
            return
        for pat in ATTN_FALLBACK_PATTERNS:
            if re.search(pat, msg):
                self.hits += 1
                # Keep it short but useful
                self.lines.append(msg.strip())
                return

warning_sniffer = WarningSniffer()

def _install_warning_hooks():
    # Hook Python warnings
    def _showwarning(message, category, filename, lineno, file=None, line=None):
        s = warnings.formatwarning(message, category, filename, lineno, line)
        warning_sniffer.observe(s)
    warnings.showwarning = _showwarning

_install_warning_hooks()

# ---------------------------
# Formatting helpers
# ---------------------------

def fmt(val, unit="s"):
    if val is None:
        return "   N/A"
    if unit == "ms":
        return f"{val*1000:7.1f}ms"
    return f"{val:7.3f}{unit}"

def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None

# ---------------------------
# Core runner
# ---------------------------

def run_stream(
    model,
    text: str,
    lang: str = "en",
    label: str = "",
    save_wav: bool = False,
    wav_run_idx: int = 1,
    **kwargs
) -> RunResult:
    """
    Run one streaming generation and return metrics.
    """
    warning_sniffer.reset()

    _cuda_sync()
    t0 = time.time()

    gen = model.generate_stream(
        text=text,
        language_id=lang,
        max_new_tokens=MAX_NEW_TOKENS,
        print_metrics=False,
        **kwargs,
    )

    chunks = []
    last_metrics = None
    for chunk, metrics in gen:
        chunks.append(chunk.squeeze(0).detach().cpu().numpy())
        last_metrics = metrics

    _cuda_sync()
    wall_time = time.time() - t0

    audio_dur = float(last_metrics.total_audio_duration) if last_metrics else 0.0

    if save_wav and chunks:
        audio = np.concatenate(chunks, axis=0)
        out = f"profile_{label}_run{wav_run_idx}.wav"
        sf.write(out, audio, model.sr)

    # Estimate vocoder vs non-vocoder time (if we ran a vocoder breakdown earlier)
    voc_total = getattr(last_metrics, "_vocoder_total_s", None) if last_metrics else None
    implied_non_vocoder = (wall_time - voc_total) if (voc_total is not None) else None

    return RunResult(
        label=label,
        chunks=len(chunks),
        first_chunk_s=getattr(last_metrics, "latency_to_first_chunk", None) if last_metrics else None,
        first_token_s=getattr(last_metrics, "first_token_time", None) if last_metrics else None,
        first_decode_s=getattr(last_metrics, "first_decode_time", None) if last_metrics else None,
        total_gen_s=getattr(last_metrics, "total_generation_time", None) if last_metrics else None,
        audio_dur_s=audio_dur,
        rtf=getattr(last_metrics, "rtf", None) if last_metrics else None,
        wall_s=wall_time,
        attn_fallback_hits=warning_sniffer.hits,
        attn_fallback_lines=warning_sniffer.lines[:6],  # cap noise
        vocoder_total_s=voc_total,
        implied_non_vocoder_s=implied_non_vocoder,
    )

# ---------------------------
# Phase 1: vocoder breakdown
# ---------------------------

def profile_vocoder_breakdown(model, text: str, lang: str = "en", **gen_kwargs):
    """
    Monkey-patch _process_token_buffer to time:
      FLOW (GPU), CROP (CPU), HIFIGAN (GPU), CPU_COPY (CPU), WATERMARK (CPU), OVERHEAD(*)
    Also stores per-run vocoder_total on metrics as _vocoder_total_s (sum of measured decode parts).
    """
    import types
    from chatterbox.models.s3tokenizer import drop_invalid_tokens

    timings = {
        "flow": [],
        "crop": [],
        "hifigan": [],
        "cpu_copy": [],
        "watermark": [],
        "overhead": [],
        "total": [],
    }

    original_method = model._process_token_buffer

    def profiled_process(
        self,
        token_buffer,
        all_tokens_so_far,
        context_window,
        start_time,
        metrics,
        print_metrics,
        fade_duration=0.02,
        skip_watermark=False,
        n_cfm_timesteps=4,
        **kw
    ):
        t_all = time.time()

        new_tokens = torch.cat(token_buffer, dim=-1)

        if all_tokens_so_far is not None and all_tokens_so_far.numel() > 0:
            context_tokens = (
                all_tokens_so_far[-context_window:]
                if all_tokens_so_far.numel() > context_window
                else all_tokens_so_far
            )
            tokens_to_process = torch.cat([context_tokens, new_tokens], dim=-1)
            context_length = int(context_tokens.numel())
        else:
            tokens_to_process = new_tokens
            context_length = 0

        clean_tokens = drop_invalid_tokens(tokens_to_process).to(self.device)
        if clean_tokens.numel() == 0:
            return None, 0.0, False

        autocast_device = "cuda" if "cuda" in str(self.device) else str(self.device)
        autocast_enabled = autocast_device in ("cuda", "mps")

        # --- FLOW (GPU) ---
        _cuda_sync()
        t = time.time()
        with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=autocast_enabled):
            output_mels = self.s3gen.flow_inference(
                speech_tokens=clean_tokens,
                ref_dict=self.conds.gen,
                finalize=True,
                n_cfm_timesteps=n_cfm_timesteps,
            )
        _cuda_sync()
        flow_t = time.time() - t
        timings["flow"].append(flow_t)

        # --- CROP (CPU) ---
        t = time.time()
        if context_length > 0:
            total_tokens = int(clean_tokens.numel())
            total_mel_frames = int(output_mels.shape[-1])
            mel_per_token = total_mel_frames / max(total_tokens, 1)
            skip_mel_frames = int(context_length * mel_per_token)
            new_mels = output_mels[:, :, skip_mel_frames:]
        else:
            new_mels = output_mels
        crop_t = time.time() - t
        timings["crop"].append(crop_t)

        if new_mels.shape[-1] == 0:
            return None, 0.0, False

        # --- HIFIGAN (GPU) ---
        _cuda_sync()
        t = time.time()
        with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=autocast_enabled):
            wav, _ = self.s3gen.hift_inference(new_mels)
        _cuda_sync()
        hifi_t = time.time() - t
        timings["hifigan"].append(hifi_t)

        # --- CPU COPY ---
        t = time.time()
        audio_chunk = wav.squeeze(0).detach().float().cpu().numpy()
        cpu_copy_t = time.time() - t
        timings["cpu_copy"].append(cpu_copy_t)

        if len(audio_chunk) == 0:
            return None, 0.0, False

        # trim_fade + edge fade
        if not hasattr(self, "_trim_fade_np"):
            self._trim_fade_np = self.s3gen.trim_fade.cpu().numpy()
        fade_len = min(len(self._trim_fade_np), len(audio_chunk))
        if metrics.chunk_count == 0 and fade_len > 0:
            audio_chunk[:fade_len] *= self._trim_fade_np[:fade_len]

        fade_samples = int(fade_duration * self.sr)
        if fade_samples > 0 and metrics.chunk_count > 0:
            fade_samples = min(fade_samples, len(audio_chunk))
            fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=audio_chunk.dtype)
            audio_chunk[:fade_samples] *= fade_in

        audio_duration = len(audio_chunk) / float(self.sr)

        # --- WATERMARK (CPU) ---
        t = time.time()
        if not skip_watermark:
            audio_chunk = self.watermarker.apply_watermark(audio_chunk, sample_rate=self.sr)
        wm_t = time.time() - t
        timings["watermark"].append(wm_t)

        audio_tensor = torch.from_numpy(audio_chunk).unsqueeze(0)

        total_t = time.time() - t_all
        overhead_t = total_t - (flow_t + crop_t + hifi_t + cpu_copy_t + wm_t)
        timings["overhead"].append(max(0.0, overhead_t))
        timings["total"].append(total_t)

        # stash a per-run accumulator on metrics so run_stream can compute implied non-vocoder
        prev = getattr(metrics, "_vocoder_total_s", 0.0)
        metrics._vocoder_total_s = float(prev) + float(total_t)

        if metrics.chunk_count == 0:
            metrics.latency_to_first_chunk = time.time() - start_time
            metrics.first_decode_time = total_t

        metrics.chunk_count += 1
        return audio_tensor, audio_duration, True

    model._process_token_buffer = types.MethodType(profiled_process, model)

    # Run once
    gen = model.generate_stream(
        text=text,
        language_id=lang,
        max_new_tokens=MAX_NEW_TOKENS,
        print_metrics=False,
        **gen_kwargs,
    )
    last_metrics = None
    for _, metrics in gen:
        last_metrics = metrics

    # Restore
    model._process_token_buffer = original_method

    return timings, last_metrics

def summarize_breakdown(timings: Dict[str, List[float]]):
    keys = ["flow", "hifigan", "crop", "cpu_copy", "watermark", "overhead", "total"]
    totals = {k: float(np.sum(timings[k])) for k in keys}
    denom = max(totals["total"], 1e-9)

    print(f"\nPer-chunk breakdown (averaged over {len(timings['total'])} chunks):")
    def line(name, k, gpu=False):
        vals = timings[k]
        if not vals:
            return
        avg = float(np.mean(vals))
        tot = float(np.sum(vals))
        pct = 100.0 * tot / denom
        tag = "(GPU)" if gpu else "(CPU)"
        print(f"  {name:>14s}{tag}: avg={avg*1000:7.1f}ms  total={tot*1000:7.1f}ms  ({pct:5.1f}%)")

    line("FLOW", "flow", gpu=True)
    line("HIFIGAN", "hifigan", gpu=True)
    line("CROP", "crop", gpu=False)
    line("CPU_COPY", "cpu_copy", gpu=False)
    line("WATERMARK", "watermark", gpu=False)
    print(f"  {'OVERHEAD':>14s}( * ): avg={np.mean(timings['overhead'])*1000:7.1f}ms  total={np.sum(timings['overhead'])*1000:7.1f}ms  ({100*np.sum(timings['overhead'])/denom:5.1f}%)")
    print(f"  {'TOTAL':>14s}: avg={np.mean(timings['total'])*1000:7.1f}ms  total={np.sum(timings['total'])*1000:7.1f}ms  (100.0%)")
    print("  (*) OVERHEAD includes python overhead, tensor bookkeeping, fades, etc.")

# ---------------------------
# Sweep helper
# ---------------------------

def run_config_median(model, cfg_label: str, cfg_kwargs: Dict[str, Any]) -> Tuple[RunResult, Dict[str, float]]:
    """
    1 discard + TIMED_RUNS timed runs -> median summary.
    Returns:
      - a representative RunResult (the median by wall time)
      - dict of medians for key metrics
    """
    # Discard runs
    for _ in range(DISCARD_RUNS):
        _ = run_stream(model, TEXT, lang=LANG, label="discard", save_wav=False, **cfg_kwargs)

    runs: List[RunResult] = []
    for k in range(1, TIMED_RUNS + 1):
        r = run_stream(
            model,
            TEXT,
            lang=LANG,
            label=cfg_label,
            save_wav=SAVE_WAV,
            wav_run_idx=k,
            **cfg_kwargs,
        )
        runs.append(r)

    # Choose representative run: median by wall time
    walls = [r.wall_s for r in runs]
    med_wall = statistics.median(walls)
    rep = min(runs, key=lambda r: abs(r.wall_s - med_wall))

    med = {
        "first_chunk_s": median([r.first_chunk_s for r in runs]),
        "rtf": median([r.rtf for r in runs]),
        "total_gen_s": median([r.total_gen_s for r in runs]),
        "audio_dur_s": median([r.audio_dur_s for r in runs]),
        "wall_s": median([r.wall_s for r in runs]),
        "attn_fallback_hits": median([r.attn_fallback_hits for r in runs]),
        "vocoder_total_s": median([r.vocoder_total_s for r in runs]),
        "implied_non_vocoder_s": median([r.implied_non_vocoder_s for r in runs]),
    }
    return rep, med

# ---------------------------
# Main
# ---------------------------

def main():
    print("=" * 70)
    print("Loading model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
    print("=" * 70)

    print("\nWarmup...")
    _ = run_stream(model, "Hello warmup.", lang=LANG, label="warmup", save_wav=False)
    print("Warmup done.\n")

    # ------------------------------------------------
    # PHASE 1: Vocoder breakdown
    # ------------------------------------------------
    print("=" * 70)
    print("PHASE 1: Vocoder time breakdown")
    print("=" * 70)

    # Note: keep watermark off during profiling unless you specifically care about it
    timings, m = profile_vocoder_breakdown(
        model,
        TEXT,
        lang=LANG,
        skip_watermark=True,
        # You can also pass n_cfm_timesteps, context_window, etc here if you want
    )
    summarize_breakdown(timings)

    # ------------------------------------------------
    # PHASE 2: Sweep
    # ------------------------------------------------
    print("\n" + "=" * 70)
    print(f"PHASE 2: Parameter sweep (median of {TIMED_RUNS} runs, after {DISCARD_RUNS} discard)")
    print("=" * 70)

    # Baseline matches your current defaults (you can tune here)
    configs = [
        {
            "label": "baseline (cfm=4, ctx=25, chunk=25, first=5, wm=off)",
            "kwargs": {
                "n_cfm_timesteps": 4,
                "context_window": 25,
                "chunk_size": 25,
                "first_chunk_size": 5,
                "skip_watermark": True,
            },
        },
        {
            "label": "first_chunk=3",
            "kwargs": {"first_chunk_size": 3, "skip_watermark": True},
        },
        {
            "label": "first_chunk=8",
            "kwargs": {"first_chunk_size": 8, "skip_watermark": True},
        },
        {
            "label": "first_chunk=12",
            "kwargs": {"first_chunk_size": 12, "skip_watermark": True},
        },
        {
            "label": "cfm=3",
            "kwargs": {"n_cfm_timesteps": 3, "skip_watermark": True},
        },
        {
            "label": "cfm=2",
            "kwargs": {"n_cfm_timesteps": 2, "skip_watermark": True},
        },
        {
            "label": "ctx=15",
            "kwargs": {"context_window": 15, "skip_watermark": True},
        },
        {
            "label": "ctx=10",
            "kwargs": {"context_window": 10, "skip_watermark": True},
        },
        {
            "label": "chunk=35, ctx=15",
            "kwargs": {"chunk_size": 35, "context_window": 15, "skip_watermark": True},
        },
        {
            "label": "chunk=40, ctx=20",
            "kwargs": {"chunk_size": 40, "context_window": 20, "skip_watermark": True},
        },
        {
            "label": "BEST CAND: cfm=3, ctx=15, chunk=35, first=5",
            "kwargs": {
                "n_cfm_timesteps": 3,
                "context_window": 15,
                "chunk_size": 35,
                "first_chunk_size": 5,
                "skip_watermark": True,
            },
        },
        {
            "label": "AGGR CAND: cfm=2, ctx=10, chunk=40, first=5",
            "kwargs": {
                "n_cfm_timesteps": 2,
                "context_window": 10,
                "chunk_size": 40,
                "first_chunk_size": 5,
                "skip_watermark": True,
            },
        },
    ]

    results = []
    for cfg in configs:
        rep, med = run_config_median(model, cfg["label"], cfg["kwargs"])
        results.append((cfg["label"], med, rep))

        print(f"\n  [{cfg['label']}]")
        print(f"    first_chunk: {fmt(med['first_chunk_s'])}  |  RTF: {fmt(med['rtf'], 'x')}  |  total: {fmt(med['total_gen_s'])}")
        print(f"    audio:       {fmt(med['audio_dur_s'])}  |  wall: {fmt(med['wall_s'])}  |  chunks: {rep.chunks}")

        # Attention fallback hints
        if rep.attn_fallback_hits:
            print(f"    ⚠️  attention-related warnings matched: {int(med['attn_fallback_hits'] or 0)} (median)")
            for line in rep.attn_fallback_lines:
                print("       - " + " ".join(line.split())[:160])

        # If we had vocoder totals attached, show implied non-vocoder
        if med.get("vocoder_total_s") is not None:
            print(f"    vocoder_total (est): {fmt(med['vocoder_total_s'])}  |  implied_non_vocoder: {fmt(med['implied_non_vocoder_s'])}")

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Config':<54s}  {'1st chunk':>9s}  {'RTF':>7s}  {'Total':>7s}  {'Wall':>7s}")
    print("-" * 70)
    for label, med, _rep in results:
        fc = f"{med['first_chunk_s']:.3f}s" if med["first_chunk_s"] is not None else "N/A"
        rtf = f"{med['rtf']:.3f}x" if med["rtf"] is not None else "N/A"
        tot = f"{med['total_gen_s']:.3f}s" if med["total_gen_s"] is not None else "N/A"
        wall = f"{med['wall_s']:.3f}s" if med["wall_s"] is not None else "N/A"
        print(f"  {label:<54s}  {fc:>9s}  {rtf:>7s}  {tot:>7s}  {wall:>7s}")
    print("=" * 70)

    if SAVE_WAV:
        print("\nListen to the generated WAVs (profile_<label>_run*.wav) to check quality,")
        print("especially for low n_cfm_timesteps and low ctx settings (quality cliffs possible).")

    print("\nNOTE:")
    print("- If you see the SDPA->manual attention fallback warnings, your T3 path is likely paying a big speed penalty.")
    print("- The chatterbox-streaming README reports ~0.472s first chunk latency and RTF ~0.499 on an RTX 4090. (Different GPU, kernel paths, and settings matter.)")

if __name__ == "__main__":
    main()