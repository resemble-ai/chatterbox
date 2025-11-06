#!/usr/bin/env python3
"""
Benchmark script to measure TTS inference performance
"""
import time
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

def benchmark_inference(model, text, num_runs=5, warmup=2):
    """Benchmark inference with warmup"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {text[:50]}...")
    print(f"{'='*60}")

    # Warmup
    print(f"Warming up ({warmup} runs)...")
    for _ in range(warmup):
        _ = model.generate(text)

    # Benchmark
    print(f"Running benchmark ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        wav = model.generate(text)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        # Calculate audio duration
        audio_duration = wav.shape[-1] / model.sr
        rtf = elapsed / audio_duration  # Real-time factor

        print(f"Run {i+1}: {elapsed:.3f}s | Audio: {audio_duration:.2f}s | RTF: {rtf:.2f}x")

    # Statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    avg_audio_dur = wav.shape[-1] / model.sr
    avg_rtf = avg_time / avg_audio_dur

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Avg inference time: {avg_time:.3f}s")
    print(f"  Min inference time: {min_time:.3f}s")
    print(f"  Max inference time: {max_time:.3f}s")
    print(f"  Audio duration: {avg_audio_dur:.2f}s")
    print(f"  Real-Time Factor: {avg_rtf:.2f}x")
    print(f"  Speed: {1/avg_rtf:.2f}x faster than realtime")
    print(f"{'='*60}\n")

    return {
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'audio_duration': avg_audio_dur,
        'rtf': avg_rtf,
        'wav': wav
    }

if __name__ == "__main__":
    device = "cuda"
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print("Loading model...")
    model = ChatterboxTTS.from_pretrained(device=device)

    # Test cases
    test_texts = [
        "Hello, this is a short test.",
        "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill.",
        "The quick brown fox jumps over the lazy dog. This is a longer sentence that should generate approximately ten seconds of audio content for our testing purposes."
    ]

    results = []
    for text in test_texts:
        result = benchmark_inference(model, text, num_runs=3, warmup=1)
        results.append(result)

        # Save sample
        ta.save(f"benchmark_{len(text)}_chars.wav", result['wav'], model.sr)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for i, (text, result) in enumerate(zip(test_texts, results)):
        print(f"Test {i+1} ({len(text)} chars):")
        print(f"  RTF: {result['rtf']:.2f}x | Time: {result['avg_time']:.3f}s | Audio: {result['audio_duration']:.2f}s")
