#!/usr/bin/env python3
"""
Chatterbox Latency Benchmark Script

Run this script after making changes to track progress toward < 200ms TTFA.
Results are saved to benchmark_results.json for historical tracking.

Usage:
    python benchmark_latency.py                    # Run standard benchmark
    python benchmark_latency.py --streaming        # Run streaming benchmark
    python benchmark_latency.py --trials 5         # More trials for accuracy
    python benchmark_latency.py --text "custom"    # Custom text
    python benchmark_latency.py --history          # View history
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


def run_streaming_benchmark(
    text: str, 
    language_id: str, 
    num_trials: int = 3, 
    warmup: bool = True,
    chunk_tokens: int = 5,
):
    """
    Run streaming latency benchmark with TTFA and chunk timing metrics.
    """
    device_info = get_device_info()
    device = device_info["device"]
    
    print("=" * 70)
    print("CHATTERBOX STREAMING BENCHMARK")
    print("=" * 70)
    print(f"Device: {device_info['gpu_name']} ({device})")
    print(f"PyTorch: {device_info['pytorch_version']}")
    print(f"Platform: {device_info['platform']}")
    print(f"Chunk Size: {chunk_tokens} tokens (~{chunk_tokens * 40}ms audio)")
    print()
    
    # Load model
    print("Loading model... (this may take a minute)")
    load_start = time.perf_counter()
    
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from chatterbox.streaming import ChatterboxStreamer
    
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    streamer = ChatterboxStreamer(model, chunk_tokens=chunk_tokens)
    
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.1f}s")
    print()
    
    print(f"Test text: {text}")
    print(f"Language: {language_id}")
    print()
    
    # Warm-up run (streaming)
    if warmup:
        print("Warm-up run (streaming)...")
        chunks = list(streamer.generate(text, language_id=language_id))
        print(f"Warm-up complete: {len(chunks)} chunks")
        print()
    
    # Measured runs
    print(f"Running {num_trials} streaming trials...")
    results = []
    
    for i in range(num_trials):
        start = time.perf_counter()
        
        chunk_times = []
        ttfa = None
        total_audio_ms = 0
        num_chunks = 0
        total_tokens = 0
        
        for chunk in streamer.generate(text, language_id=language_id):
            chunk_time = (time.perf_counter() - start) * 1000
            
            if ttfa is None:
                ttfa = chunk_time
            
            chunk_times.append({
                "chunk_idx": chunk.chunk_index,
                "latency_ms": round(chunk_time, 1),
                "audio_duration_ms": round(chunk.audio_duration_ms, 1),
                "num_tokens": chunk.num_tokens,
            })
            
            total_audio_ms += chunk.audio_duration_ms
            num_chunks += 1
            total_tokens = chunk.total_tokens
        
        total_time = (time.perf_counter() - start) * 1000
        
        trial_result = {
            "trial": i + 1,
            "ttfa_ms": round(ttfa, 1) if ttfa else 0,
            "total_latency_ms": round(total_time, 1),
            "total_audio_ms": round(total_audio_ms, 1),
            "num_chunks": num_chunks,
            "total_tokens": total_tokens,
            "chunk_details": chunk_times,
        }
        results.append(trial_result)
        
        print(f"  Trial {i+1}: TTFA={ttfa:.0f}ms, Total={total_time:.0f}ms, "
              f"Chunks={num_chunks}, Audio={total_audio_ms:.0f}ms")
    
    # Calculate statistics
    ttfa_values = [r["ttfa_ms"] for r in results]
    total_latencies = [r["total_latency_ms"] for r in results]
    
    avg_ttfa = sum(ttfa_values) / len(ttfa_values)
    min_ttfa = min(ttfa_values)
    max_ttfa = max(ttfa_values)
    
    avg_total = sum(total_latencies) / len(total_latencies)
    avg_audio = sum(r["total_audio_ms"] for r in results) / len(results)
    avg_chunks = sum(r["num_chunks"] for r in results) / len(results)
    
    rtf = avg_total / avg_audio if avg_audio > 0 else 0
    
    # Summary
    print()
    print("=" * 70)
    print("STREAMING RESULTS")
    print("=" * 70)
    print(f"Average TTFA:     {avg_ttfa:.0f}ms (Time to First Audio)")
    print(f"Best TTFA:        {min_ttfa:.0f}ms")
    print(f"Worst TTFA:       {max_ttfa:.0f}ms")
    print(f"Total Latency:    {avg_total:.0f}ms (avg)")
    print(f"Audio Duration:   {avg_audio:.0f}ms (avg)")
    print(f"Avg Chunks:       {avg_chunks:.1f}")
    print(f"Real-time Factor: {rtf:.2f}x {'(slower than real-time)' if rtf > 1 else '(faster than real-time)'}")
    print()
    print(f"TARGET: < 200ms TTFA for conversational AI")
    
    ttfa_gap = avg_ttfa - 200
    if ttfa_gap > 0:
        print(f"TTFA GAP: {ttfa_gap:.0f}ms to close ❌")
    else:
        print(f"TTFA TARGET MET! {-ttfa_gap:.0f}ms under target ✅")
    print("=" * 70)
    
    # Compile full results
    benchmark_result = {
        "timestamp": datetime.now().isoformat(),
        "mode": "streaming",
        "device_info": device_info,
        "test_config": {
            "text": text,
            "language_id": language_id,
            "num_trials": num_trials,
            "chunk_tokens": chunk_tokens,
        },
        "trials": results,
        "summary": {
            "avg_ttfa_ms": round(avg_ttfa, 1),
            "min_ttfa_ms": round(min_ttfa, 1),
            "max_ttfa_ms": round(max_ttfa, 1),
            "avg_total_latency_ms": round(avg_total, 1),
            "avg_audio_duration_ms": round(avg_audio, 1),
            "avg_num_chunks": round(avg_chunks, 1),
            "real_time_factor": round(rtf, 2),
            "ttfa_target_met": ttfa_gap <= 0,
            "ttfa_gap_ms": round(ttfa_gap, 1),
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
    
    print("=" * 95)
    print("BENCHMARK HISTORY")
    print("=" * 95)
    print(f"{'#':<3} {'Date':<20} {'Mode':<10} {'Device':<10} {'Avg/TTFA':<12} {'Best':<10} {'Target':<8}")
    print("-" * 95)
    
    for i, result in enumerate(history, 1):
        ts = result.get("timestamp", "Unknown")[:19].replace("T", " ")
        device = result.get("device_info", {}).get("device", "?")
        mode = result.get("mode", "standard")
        summary = result.get("summary", {})
        
        # Handle both streaming and standard results
        if mode == "streaming":
            avg = summary.get("avg_ttfa_ms", 0)
            best = summary.get("min_ttfa_ms", 0)
            met = "✅" if summary.get("ttfa_target_met", False) else "❌"
            mode_str = "stream"
        else:
            avg = summary.get("avg_latency_ms", 0)
            best = summary.get("min_latency_ms", 0)
            met = "✅" if summary.get("target_met", False) else "❌"
            mode_str = "batch"
        
        print(f"{i:<3} {ts:<20} {mode_str:<10} {device:<10} {avg:>8.0f}ms   {best:>6.0f}ms   {met}")
    
    print("-" * 95)
    
    # Show improvement over time (comparing same mode)
    standard_results = [r for r in history if r.get("mode", "standard") != "streaming"]
    streaming_results = [r for r in history if r.get("mode") == "streaming"]
    
    if len(standard_results) >= 2:
        first = standard_results[0]["summary"]["avg_latency_ms"]
        last = standard_results[-1]["summary"]["avg_latency_ms"]
        improvement = first - last
        pct = (improvement / first) * 100 if first > 0 else 0
        print(f"\nBatch Progress: {first:.0f}ms → {last:.0f}ms ({improvement:+.0f}ms, {pct:+.1f}%)")
    
    if len(streaming_results) >= 2:
        first = streaming_results[0]["summary"]["avg_ttfa_ms"]
        last = streaming_results[-1]["summary"]["avg_ttfa_ms"]
        improvement = first - last
        pct = (improvement / first) * 100 if first > 0 else 0
        print(f"TTFA Progress: {first:.0f}ms → {last:.0f}ms ({improvement:+.0f}ms, {pct:+.1f}%)")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Chatterbox TTS latency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_latency.py                     # Default (non-streaming) benchmark
  python benchmark_latency.py --streaming         # Streaming benchmark with TTFA
  python benchmark_latency.py --streaming --chunk-tokens 3  # Smaller chunks
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
        "--streaming", "-s",
        action="store_true",
        help="Run streaming benchmark with TTFA metrics"
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=5,
        help="Tokens per chunk for streaming mode (default: 5, ~200ms audio)"
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
    
    # Run benchmark (streaming or standard)
    if args.streaming:
        result = run_streaming_benchmark(
            text=args.text,
            language_id=args.lang,
            num_trials=args.trials,
            warmup=not args.no_warmup,
            chunk_tokens=args.chunk_tokens,
        )
    else:
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

