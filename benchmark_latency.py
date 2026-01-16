#!/usr/bin/env python3
"""
Chatterbox Latency Benchmark Script

Run this script after making changes to track progress toward < 200ms TTFA.
Results are saved to benchmark_results.json for historical tracking.

Usage:
    python benchmark_latency.py                    # Run benchmark
    python benchmark_latency.py --trials 5         # Run 5 trials
    python benchmark_latency.py --text "custom"    # Custom text
    python benchmark_latency.py --history          # Show history
"""

import argparse
import json
import time
import platform
import sys
from datetime import datetime
from pathlib import Path

import torch

# Results file for tracking progress
RESULTS_FILE = Path(__file__).parent / "benchmark_results.json"


def get_device_info():
    """Gather device and environment information."""
    info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
    }
    
    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        info["cuda_version"] = torch.version.cuda
    elif torch.backends.mps.is_available():
        info["device"] = "mps"
        info["gpu_name"] = "Apple Silicon (MPS)"
    else:
        info["device"] = "cpu"
        info["gpu_name"] = "None"
    
    return info


def run_benchmark(text: str, language_id: str, num_trials: int = 3, warmup: bool = True):
    """
    Run latency benchmark and return detailed results.
    """
    device_info = get_device_info()
    device = device_info["device"]
    
    print("=" * 70)
    print("CHATTERBOX LATENCY BENCHMARK")
    print("=" * 70)
    print(f"Device: {device_info['gpu_name']} ({device})")
    print(f"PyTorch: {device_info['pytorch_version']}")
    print(f"Platform: {device_info['platform']}")
    print()
    
    # Load model
    print("Loading model... (this may take a minute)")
    load_start = time.perf_counter()
    
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.1f}s")
    print()
    
    print(f"Test text: {text}")
    print(f"Language: {language_id}")
    print()
    
    # Warm-up run
    if warmup:
        print("Warm-up run...")
        _ = model.generate(text, language_id=language_id)
        print("Warm-up complete")
        print()
    
    # Measured runs
    print(f"Running {num_trials} measured trials...")
    results = []
    
    for i in range(num_trials):
        start = time.perf_counter()
        wav = model.generate(text, language_id=language_id)
        latency_ms = (time.perf_counter() - start) * 1000
        
        audio_samples = wav.shape[-1]
        audio_duration_ms = audio_samples / 22050 * 1000
        
        trial_result = {
            "trial": i + 1,
            "latency_ms": round(latency_ms, 1),
            "audio_duration_ms": round(audio_duration_ms, 1),
            "audio_samples": audio_samples,
        }
        results.append(trial_result)
        
        print(f"  Trial {i+1}: {latency_ms:.0f}ms (audio: {audio_duration_ms:.0f}ms)")
    
    # Calculate statistics
    latencies = [r["latency_ms"] for r in results]
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    
    avg_audio = sum(r["audio_duration_ms"] for r in results) / len(results)
    rtf = avg_latency / avg_audio if avg_audio > 0 else 0
    
    # Summary
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Average Latency:  {avg_latency:.0f}ms")
    print(f"Best Latency:     {min_latency:.0f}ms")
    print(f"Worst Latency:    {max_latency:.0f}ms")
    print(f"Audio Duration:   {avg_audio:.0f}ms (avg)")
    print(f"Real-time Factor: {rtf:.2f}x {'(slower than real-time)' if rtf > 1 else '(faster than real-time)'}")
    print()
    print(f"TARGET: < 200ms TTFA for conversational AI")
    
    gap = avg_latency - 200
    if gap > 0:
        print(f"GAP: {gap:.0f}ms to close ❌")
    else:
        print(f"TARGET MET! {-gap:.0f}ms under target ✅")
    print("=" * 70)
    
    # Compile full results
    benchmark_result = {
        "timestamp": datetime.now().isoformat(),
        "device_info": device_info,
        "test_config": {
            "text": text,
            "language_id": language_id,
            "num_trials": num_trials,
        },
        "trials": results,
        "summary": {
            "avg_latency_ms": round(avg_latency, 1),
            "min_latency_ms": round(min_latency, 1),
            "max_latency_ms": round(max_latency, 1),
            "avg_audio_duration_ms": round(avg_audio, 1),
            "real_time_factor": round(rtf, 2),
            "target_met": gap <= 0,
            "gap_ms": round(gap, 1),
        },
    }
    
    return benchmark_result


def save_result(result: dict):
    """Save benchmark result to history file."""
    history = []
    
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, "r") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            history = []
    
    history.append(result)
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"Total benchmarks in history: {len(history)}")


def show_history():
    """Display benchmark history."""
    if not RESULTS_FILE.exists():
        print("No benchmark history found. Run a benchmark first!")
        return
    
    with open(RESULTS_FILE, "r") as f:
        history = json.load(f)
    
    if not history:
        print("No benchmark history found.")
        return
    
    print("=" * 80)
    print("BENCHMARK HISTORY")
    print("=" * 80)
    print(f"{'#':<3} {'Date':<20} {'Device':<15} {'Avg Latency':<12} {'Best':<10} {'Target':<8}")
    print("-" * 80)
    
    for i, result in enumerate(history, 1):
        ts = result.get("timestamp", "Unknown")[:19].replace("T", " ")
        device = result.get("device_info", {}).get("device", "?")
        summary = result.get("summary", {})
        avg = summary.get("avg_latency_ms", 0)
        best = summary.get("min_latency_ms", 0)
        met = "✅" if summary.get("target_met", False) else "❌"
        
        print(f"{i:<3} {ts:<20} {device:<15} {avg:>8.0f}ms   {best:>6.0f}ms   {met}")
    
    print("-" * 80)
    
    # Show improvement over time
    if len(history) >= 2:
        first = history[0]["summary"]["avg_latency_ms"]
        last = history[-1]["summary"]["avg_latency_ms"]
        improvement = first - last
        pct = (improvement / first) * 100 if first > 0 else 0
        
        print(f"\nProgress: {first:.0f}ms → {last:.0f}ms ({improvement:+.0f}ms, {pct:+.1f}%)")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Chatterbox TTS latency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_latency.py                     # Default benchmark
  python benchmark_latency.py --trials 5          # More trials for accuracy
  python benchmark_latency.py --text "مرحبا"      # Custom Arabic text
  python benchmark_latency.py --lang en           # Test English
  python benchmark_latency.py --history           # View history
  python benchmark_latency.py --no-save           # Don't save to history
        """
    )
    
    parser.add_argument(
        "--text", "-t",
        default="مرحباً، كيف حالك اليوم؟",
        help="Text to synthesize (default: Arabic greeting)"
    )
    parser.add_argument(
        "--lang", "-l",
        default="ar",
        help="Language ID (default: ar)"
    )
    parser.add_argument(
        "--trials", "-n",
        type=int,
        default=3,
        help="Number of trials (default: 3)"
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warm-up run"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to history"
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show benchmark history and exit"
    )
    
    args = parser.parse_args()
    
    if args.history:
        show_history()
        return
    
    # Run benchmark
    result = run_benchmark(
        text=args.text,
        language_id=args.lang,
        num_trials=args.trials,
        warmup=not args.no_warmup,
    )
    
    # Save result
    if not args.no_save:
        save_result(result)
    
    print("\nRun 'python benchmark_latency.py --history' to see all results")


if __name__ == "__main__":
    main()

