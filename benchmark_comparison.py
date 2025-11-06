#!/usr/bin/env python3
"""
Benchmark comparison between original and optimized TTS

This script measures the performance difference between the baseline
and optimized implementations.
"""
import time
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from chatterbox.optimized_tts import OptimizedChatterboxTTS


def benchmark_model(model, text, name="Model", num_runs=5, warmup=2, verbose=False):
    """Benchmark a model with warmup"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"Text: {text[:60]}...")
    print(f"{'='*60}")

    # Warmup
    print(f"Warming up ({warmup} runs)...")
    for _ in range(warmup):
        _ = model.generate(text, verbose=False) if hasattr(model, 'generate') else model.generate(text)

    # Benchmark
    print(f"Running benchmark ({num_runs} runs)...")
    times = []
    audio_durations = []

    for i in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        if hasattr(model, 'generate'):
            wav = model.generate(text, verbose=verbose)
        else:
            wav = model.generate(text)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)

        # Calculate audio duration
        audio_duration = wav.shape[-1] / model.sr
        audio_durations.append(audio_duration)

        rtf = elapsed / audio_duration  # Real-time factor
        print(f"  Run {i+1}: {elapsed:.3f}s | Audio: {audio_duration:.2f}s | RTF: {rtf:.2f}x | Speed: {1/rtf:.2f}x realtime")

    # Statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    avg_audio_dur = sum(audio_durations) / len(audio_durations)
    avg_rtf = avg_time / avg_audio_dur

    print(f"\n{'='*60}")
    print(f"Results for {name}:")
    print(f"  Avg inference time: {avg_time:.3f}s ± {max(times) - min(times):.3f}s")
    print(f"  Audio duration: {avg_audio_dur:.2f}s")
    print(f"  Real-Time Factor: {avg_rtf:.2f}x")
    print(f"  Speed: {1/avg_rtf:.2f}x faster than realtime")
    print(f"{'='*60}\n")

    return {
        'name': name,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'audio_duration': avg_audio_dur,
        'rtf': avg_rtf,
        'speedup_vs_realtime': 1/avg_rtf,
        'wav': wav,
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"CHATTERBOX TTS PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}")

    # Test texts
    test_cases = [
        {
            'name': 'Short (5s)',
            'text': 'Hello, this is a short test sentence.',
        },
        {
            'name': 'Medium (10s)',
            'text': 'Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy\'s Nexus in an epic late-game pentakill.',
        },
        {
            'name': 'Long (15s)',
            'text': 'The quick brown fox jumps over the lazy dog. This is a longer sentence that should generate approximately fifteen seconds of audio content for our comprehensive testing and benchmarking purposes today.',
        },
    ]

    for test_case in test_cases:
        print(f"\n\n{'#'*70}")
        print(f"TEST CASE: {test_case['name']}")
        print(f"{'#'*70}")

        # Load baseline model
        print("\n📦 Loading BASELINE model...")
        baseline_model = ChatterboxTTS.from_pretrained(device=device)
        print("✅ Baseline model loaded")

        # Benchmark baseline
        baseline_result = benchmark_model(
            baseline_model,
            test_case['text'],
            name=f"Baseline - {test_case['name']}",
            num_runs=3,
            warmup=1,
            verbose=False,
        )

        # Save baseline audio
        ta.save(
            f"baseline_{test_case['name'].lower().replace(' ', '_')}.wav",
            baseline_result['wav'],
            baseline_model.sr
        )

        # Clear memory
        del baseline_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load optimized model
        print("\n🚀 Loading OPTIMIZED model...")
        optimized_model = OptimizedChatterboxTTS.from_pretrained(
            device=device,
            enable_compilation=True,
            use_mixed_precision=True,
            enable_watermark=False,  # Disable for max speed
        )
        print("✅ Optimized model loaded")

        # Benchmark optimized
        optimized_result = benchmark_model(
            optimized_model,
            test_case['text'],
            name=f"Optimized - {test_case['name']}",
            num_runs=3,
            warmup=1,
            verbose=False,
        )

        # Save optimized audio
        ta.save(
            f"optimized_{test_case['name'].lower().replace(' ', '_')}.wav",
            optimized_result['wav'],
            optimized_model.sr
        )

        # Calculate speedup
        speedup = baseline_result['avg_time'] / optimized_result['avg_time']
        rtf_improvement = baseline_result['rtf'] / optimized_result['rtf']

        print(f"\n{'='*70}")
        print(f"COMPARISON - {test_case['name']}")
        print(f"{'='*70}")
        print(f"Baseline:  {baseline_result['avg_time']:.3f}s (RTF: {baseline_result['rtf']:.2f}x)")
        print(f"Optimized: {optimized_result['avg_time']:.3f}s (RTF: {optimized_result['rtf']:.2f}x)")
        print(f"\n🎯 SPEEDUP: {speedup:.2f}x faster")
        print(f"🎯 RTF IMPROVEMENT: {rtf_improvement:.2f}x")
        print(f"{'='*70}")

        # Clear memory
        del optimized_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n\n{'='*70}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print("Audio files saved for quality comparison:")
    print("  - baseline_*.wav")
    print("  - optimized_*.wav")
    print("\nNote: Listen to both files to verify quality is maintained!")
    print(f"{'='*70}\n")
