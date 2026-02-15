"""
Streaming TTS profiler — breaks down where decode time goes,
then tests parameter variations to find the fastest config.

Usage:
    python test_stream_profile.py
"""

import time
import numpy as np
import torch
import soundfile as sf
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

import torch
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

DEVICE = "cuda"
TEXT = "The weather today is absolutely beautiful, perfect for a walk in the park."

def run_stream(model, text, lang="en", label="", save_wav=False, **kwargs):
    """Run one streaming generation and return metrics."""
    # Force CUDA sync for accurate timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    t0 = time.time()
    gen = model.generate_stream(
        text=text,
        language_id=lang,
        max_new_tokens=400,
        print_metrics=False,
        **kwargs,
    )
    chunks = []
    last_metrics = None
    for chunk, metrics in gen:
        chunks.append(chunk.squeeze(0).detach().cpu().numpy())
        last_metrics = metrics

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_time = time.time() - t0

    audio_dur = last_metrics.total_audio_duration if last_metrics else 0
    if save_wav and chunks:
        audio = np.concatenate(chunks, axis=0)
        sf.write(f"profile_{label}.wav", audio, model.sr)

    return {
        "label": label,
        "chunks": len(chunks),
        "first_chunk_s": last_metrics.latency_to_first_chunk if last_metrics else None,
        "first_token_s": last_metrics.first_token_time if last_metrics else None,
        "first_decode_s": last_metrics.first_decode_time if last_metrics else None,
        "total_gen_s": last_metrics.total_generation_time if last_metrics else None,
        "audio_dur_s": audio_dur,
        "rtf": last_metrics.rtf if last_metrics else None,
        "wall_s": wall_time,
    }


def profile_vocoder_breakdown(model, text, lang="en"):
    """
    Monkey-patch _process_token_buffer to time flow vs hifigan vs watermark separately.
    """
    import types
    from chatterbox.models.s3tokenizer import drop_invalid_tokens
    
    timings = {"flow": [], "crop": [], "hifigan": [], "watermark": [], "overhead": []}
    
    original_method = model._process_token_buffer
    
    def profiled_process(self, token_buffer, all_tokens_so_far, context_window,
                         start_time, metrics, print_metrics, fade_duration=0.02,
                         skip_watermark=False, n_cfm_timesteps=4):
        t_start = time.time()
        
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

        # --- FLOW ---
        torch.cuda.synchronize()
        t_flow = time.time()
        with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=autocast_enabled):
            output_mels = self.s3gen.flow_inference(
                speech_tokens=clean_tokens,
                ref_dict=self.conds.gen,
                finalize=True,
                n_cfm_timesteps=n_cfm_timesteps,
            )
        torch.cuda.synchronize()
        timings["flow"].append(time.time() - t_flow)

        # --- CROP ---
        t_crop = time.time()
        if context_length > 0:
            total_tokens = int(clean_tokens.numel())
            total_mel_frames = output_mels.shape[-1]
            mel_per_token = total_mel_frames / max(total_tokens, 1)
            skip_mel_frames = int(context_length * mel_per_token)
            new_mels = output_mels[:, :, skip_mel_frames:]
        else:
            new_mels = output_mels

        if new_mels.shape[-1] == 0:
            return None, 0.0, False
        timings["crop"].append(time.time() - t_crop)

        # --- HIFIGAN ---
        torch.cuda.synchronize()
        t_hifi = time.time()
        with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=autocast_enabled):
            wav, _ = self.s3gen.hift_inference(new_mels)
        torch.cuda.synchronize()
        timings["hifigan"].append(time.time() - t_hifi)
        
        audio_chunk = wav.squeeze(0).detach().float().cpu().numpy()
        if len(audio_chunk) == 0:
            return None, 0.0, False

        # trim_fade + edge fade
        if not hasattr(self, '_trim_fade_np'):
            self._trim_fade_np = self.s3gen.trim_fade.cpu().numpy()
        fade_len = min(len(self._trim_fade_np), len(audio_chunk))
        if metrics.chunk_count == 0 and fade_len > 0:
            audio_chunk[:fade_len] *= self._trim_fade_np[:fade_len]

        fade_samples = int(fade_duration * self.sr)
        if fade_samples > 0 and metrics.chunk_count > 0:
            fade_samples = min(fade_samples, len(audio_chunk))
            fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=audio_chunk.dtype)
            audio_chunk[:fade_samples] *= fade_in

        audio_duration = len(audio_chunk) / self.sr

        # --- WATERMARK ---
        t_wm = time.time()
        if not skip_watermark:
            audio_chunk = self.watermarker.apply_watermark(audio_chunk, sample_rate=self.sr)
        timings["watermark"].append(time.time() - t_wm)

        audio_tensor = torch.from_numpy(audio_chunk).unsqueeze(0)

        decode_time = time.time() - t_start
        timings["overhead"].append(decode_time - timings["flow"][-1] - timings["hifigan"][-1] - timings["watermark"][-1])

        if metrics.chunk_count == 0:
            metrics.latency_to_first_chunk = time.time() - start_time
            metrics.first_decode_time = decode_time
        metrics.chunk_count += 1
        return audio_tensor, audio_duration, True

    # Bind the profiled method
    model._process_token_buffer = types.MethodType(profiled_process, model)
    
    # Run once
    gen = model.generate_stream(text=text, language_id=lang, max_new_tokens=400, print_metrics=False)
    chunks = []
    last_metrics = None
    for chunk, metrics in gen:
        chunks.append(chunk.squeeze(0).detach().cpu().numpy())
        last_metrics = metrics
    
    # Restore original
    model._process_token_buffer = original_method
    
    return timings, last_metrics


def fmt(val, unit="s"):
    if val is None:
        return "  N/A"
    return f"{val:6.3f}{unit}"

# ── Load ──
print("=" * 70)
print("Loading model...")
model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
print("=" * 70)

# ── Warmup ──
print("\nWarmup...")
run_stream(model, "Hello warmup.", label="warmup")
print("Warmup done.\n")

# ══════════════════════════════════════════════════════════════
# PHASE 1: Profile where decode time goes
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 1: Vocoder time breakdown")
print("=" * 70)

timings, m = profile_vocoder_breakdown(model, TEXT)

print(f"\nPer-chunk breakdown (averaged over {len(timings['flow'])} chunks):")
for key in ["flow", "hifigan", "watermark", "overhead"]:
    vals = timings[key]
    if vals:
        print(f"  {key:>12s}: avg={np.mean(vals)*1000:6.1f}ms  total={np.sum(vals)*1000:6.1f}ms  ({100*np.sum(vals)/sum(np.sum(timings[k]) for k in timings):4.1f}%)")

total_decode = sum(np.sum(timings[k]) for k in timings)
print(f"  {'TOTAL':>12s}: {total_decode*1000:6.1f}ms across {len(timings['flow'])} chunks")

# ══════════════════════════════════════════════════════════════
# PHASE 2: Test parameter variations
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PHASE 2: Parameter sweep")
print("=" * 70)

configs = [
    {"label": "baseline (cfm=4, ctx=25, chunk=25)",
     "kwargs": {}},
    
    {"label": "cfm=3",
     "kwargs": {"n_cfm_timesteps": 3, "skip_watermark": True}},
    
    {"label": "cfm=2",
     "kwargs": {"n_cfm_timesteps": 2, "skip_watermark": True}},
    
    {"label": "skip_watermark",
     "kwargs": {"skip_watermark": True}},
    
    {"label": "ctx=15",
     "kwargs": {"context_window": 15, "skip_watermark": True}},
    
    {"label": "ctx=10",
     "kwargs": {"context_window": 10, "skip_watermark": True}},
    
    {"label": "chunk=40, ctx=20",
     "kwargs": {"chunk_size": 40, "skip_watermark": True, "context_window": 20}},

    {"label": "BEST: cfm=3, skip_wm, ctx=15, chunk=35",
     "kwargs": {"n_cfm_timesteps": 3, "skip_watermark": True, "context_window": 15, "chunk_size": 35}},
    
    {"label": "AGGRESSIVE: cfm=2, skip_wm, ctx=10, chunk=40",
     "kwargs": {"n_cfm_timesteps": 2, "skip_watermark": True, "context_window": 10, "chunk_size": 40}},
]

results = []
for cfg in configs:
    # Run twice, take second (avoid any one-off effects)
    run_stream(model, TEXT, label="discard", **cfg["kwargs"])
    r = run_stream(model, TEXT, label=cfg["label"], save_wav=True, **cfg["kwargs"])
    results.append((cfg["label"], r))
    
    print(f"\n  [{cfg['label']}]")
    print(f"    first_chunk: {fmt(r['first_chunk_s'])}  |  RTF: {fmt(r['rtf'], 'x')}  |  total: {fmt(r['total_gen_s'])}  |  audio: {fmt(r['audio_dur_s'])}  |  chunks: {r['chunks']}")

# ── Summary table ──
print(f"\n{'=' * 70}")
print(f"{'Config':<50s}  {'1st chunk':>9s}  {'RTF':>6s}  {'Total':>7s}")
print(f"{'-' * 70}")
for label, r in results:
    fc = f"{r['first_chunk_s']:.3f}s" if r['first_chunk_s'] else "N/A"
    rtf = f"{r['rtf']:.3f}x" if r['rtf'] else "N/A"
    tot = f"{r['total_gen_s']:.3f}s" if r['total_gen_s'] else "N/A"
    print(f"  {label:<48s}  {fc:>9s}  {rtf:>6s}  {tot:>7s}")
print("=" * 70)
print("\nListen to profile_BEST*.wav and profile_AGGRESSIVE*.wav to check quality!")