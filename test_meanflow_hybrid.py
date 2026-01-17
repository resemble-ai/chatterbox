#!/usr/bin/env python3
"""
MeanFlow Hybrid Test Script

This script tests the combination of:
- Multilingual T3 model (for text → speech tokens)
- Pre-trained MeanFlow S3Gen from Chatterbox Turbo (2-step CFM instead of 10)

This is a proof-of-concept to validate that the MeanFlow model can work with
the multilingual pipeline before investing time in training an Arabic-specific version.

Expected improvement: ~5x faster S3Gen (from ~500ms to ~100ms per chunk)
"""

import os
import gc
import time
import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download


def test_meanflow_hybrid(
    text: str = "Hello, how are you today?",
    language_id: str = "en",
    trials: int = 3,
    streaming: bool = True,
):
    """Test the MeanFlow hybrid setup."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("MEANFLOW HYBRID TEST")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Text: {text}")
    print(f"Language: {language_id}")
    print(f"Streaming: {streaming}")
    print()
    
    # Step 1: Load the multilingual model
    print("Step 1: Loading multilingual model...")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    
    load_start = time.perf_counter()
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    load_time = time.perf_counter() - load_start
    print(f"  Multilingual model loaded in {load_time:.1f}s")
    
    # Save original S3Gen for comparison
    original_s3gen = model.s3gen
    original_meanflow_flag = original_s3gen.meanflow
    print(f"  Original S3Gen meanflow={original_meanflow_flag}")
    
    # Clear CUDA cache to maximize available VRAM
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        print(f"  Available VRAM: {free_mem / 1024**3:.2f} GB")
    
    # Step 2: Check if MeanFlow is available (download path only, don't load yet to save VRAM)
    print("\nStep 2: Checking MeanFlow S3Gen availability...")
    
    meanflow_path = None
    meanflow_available = False
    
    # Check for local MeanFlow model first
    local_meanflow_paths = [
        Path("./models/s3gen_meanflow.safetensors"),
        Path("./models/s3gen_meanflow.pt"),
        Path("./s3gen_meanflow.safetensors"),
        Path("./meanflow_checkpoints/s3gen_meanflow_arabic.pt"),
    ]
    
    for local_path in local_meanflow_paths:
        if local_path.exists():
            print(f"  Found local MeanFlow model at {local_path}")
            meanflow_path = local_path
            meanflow_available = True
            break
    
    # Try downloading from HuggingFace if no local model (download only, don't load)
    if not meanflow_available:
        try:
            turbo_path = snapshot_download(
                repo_id="ResembleAI/chatterbox-turbo",
                token=os.getenv("HF_TOKEN"),  # Will be None if not set
                allow_patterns=["s3gen_meanflow.safetensors"]
            )
            meanflow_path = Path(turbo_path) / "s3gen_meanflow.safetensors"
            if meanflow_path.exists():
                meanflow_available = True
                print(f"  MeanFlow S3Gen available at {meanflow_path}")
                print(f"  (Will load after baseline test to save VRAM)")
        except Exception as e:
            print(f"  ⚠️  Could not access MeanFlow model: {type(e).__name__}")
            print(f"     The Chatterbox Turbo model may be private/gated.")
            print(f"     Options:")
            print(f"       1. Request access at: https://huggingface.co/ResembleAI/chatterbox-turbo")
            print(f"       2. Set HF_TOKEN environment variable after getting access")
            print(f"       3. Train your own MeanFlow model (see ROADMAP.md Task 2.1)")
            print(f"     Continuing with baseline-only test...")
    
    # Step 3: Run baseline test with original 10-step S3Gen
    print("\n" + "=" * 70)
    print("BASELINE TEST (Original 10-step S3Gen)")
    print("=" * 70)
    
    model.s3gen = original_s3gen
    
    if streaming:
        baseline_results = run_streaming_test(model, text, language_id, trials, "Baseline")
    else:
        baseline_results = run_batch_test(model, text, language_id, trials, "Baseline")
    
    meanflow_results = None
    
    # Step 4: Run test with MeanFlow 2-step S3Gen (if available)
    if meanflow_available and meanflow_path is not None:
        print("\n" + "=" * 70)
        print("MEANFLOW TEST (2-step S3Gen)")
        print("=" * 70)
        
        # Free the original S3Gen to make room for MeanFlow
        print("  Freeing original S3Gen from VRAM...")
        model.s3gen = None
        original_s3gen.cpu()
        del original_s3gen
        torch.cuda.empty_cache()
        gc.collect()
        
        # Now load MeanFlow S3Gen
        print("  Loading MeanFlow S3Gen...")
        from chatterbox.models.s3gen import S3Gen
        meanflow_s3gen = S3Gen(meanflow=True)
        if meanflow_path.suffix == ".safetensors":
            weights = load_file(meanflow_path)
        else:
            weights = torch.load(meanflow_path, map_location=device)
        meanflow_s3gen.load_state_dict(weights, strict=True)
        meanflow_s3gen.to(device).eval()
        print(f"  MeanFlow S3Gen loaded (meanflow=True, 2-step CFM)")
        
        model.s3gen = meanflow_s3gen
        
        if streaming:
            meanflow_results = run_streaming_test(model, text, language_id, trials, "MeanFlow")
        else:
            meanflow_results = run_batch_test(model, text, language_id, trials, "MeanFlow")
    
    # Step 5: Compare results or show training instructions
    print("\n" + "=" * 70)
    
    if meanflow_results is not None:
        print("COMPARISON")
        print("=" * 70)
        
        if streaming:
            baseline_ttfa = baseline_results['avg_ttfa']
            meanflow_ttfa = meanflow_results['avg_ttfa']
            baseline_rtf = baseline_results['avg_rtf']
            meanflow_rtf = meanflow_results['avg_rtf']
            
            ttfa_improvement = (baseline_ttfa - meanflow_ttfa) / baseline_ttfa * 100
            rtf_improvement = (baseline_rtf - meanflow_rtf) / baseline_rtf * 100
            
            print(f"TTFA: {baseline_ttfa:.0f}ms → {meanflow_ttfa:.0f}ms ({ttfa_improvement:+.1f}%)")
            print(f"RTF:  {baseline_rtf:.2f}x → {meanflow_rtf:.2f}x ({rtf_improvement:+.1f}%)")
        else:
            baseline_lat = baseline_results['avg_latency']
            meanflow_lat = meanflow_results['avg_latency']
            improvement = (baseline_lat - meanflow_lat) / baseline_lat * 100
            print(f"Latency: {baseline_lat:.0f}ms → {meanflow_lat:.0f}ms ({improvement:+.1f}%)")
        
        print()
        if streaming and meanflow_ttfa < baseline_ttfa:
            print("✅ MeanFlow IMPROVES latency! Proof-of-concept successful.")
            print("   Consider training an Arabic-specific MeanFlow model for best results.")
        elif not streaming and meanflow_lat < baseline_lat:
            print("✅ MeanFlow IMPROVES latency! Proof-of-concept successful.")
        else:
            print("⚠️  Results inconclusive. May need more testing or Arabic-specific training.")
    else:
        print("MEANFLOW NOT AVAILABLE - TRAINING REQUIRED")
        print("=" * 70)
        print()
        print("The Chatterbox Turbo MeanFlow model is private/gated.")
        print("To get ~5x speedup on S3Gen, you need to train your own MeanFlow model.")
        print()
        print("📋 NEXT STEPS:")
        print("   1. See ROADMAP.md Task 2.1 for MeanFlow distillation instructions")
        print("   2. See FINE_TUNING_GUIDE.md Section 6.1 for complete training code")
        print()
        print("📊 ESTIMATED IMPROVEMENT (based on MeanFlow architecture):")
        if streaming:
            baseline_ttfa = baseline_results['avg_ttfa']
            baseline_rtf = baseline_results['avg_rtf']
            estimated_ttfa = baseline_ttfa * 0.4  # ~60% reduction from 2-step CFM
            estimated_rtf = baseline_rtf * 0.3    # ~70% reduction
            print(f"   Current TTFA: {baseline_ttfa:.0f}ms → Estimated with MeanFlow: ~{estimated_ttfa:.0f}ms")
            print(f"   Current RTF:  {baseline_rtf:.2f}x → Estimated with MeanFlow: ~{estimated_rtf:.2f}x")
        else:
            baseline_lat = baseline_results['avg_latency']
            estimated_lat = baseline_lat * 0.4
            print(f"   Current Latency: {baseline_lat:.0f}ms → Estimated with MeanFlow: ~{estimated_lat:.0f}ms")
        print()
        print("💡 TIP: MeanFlow reduces CFM denoising from 10 steps to 2 steps,")
        print("   which is the biggest single optimization for S3Gen latency.")
    
    return baseline_results, meanflow_results


def run_streaming_test(model, text, language_id, trials, label):
    """Run streaming benchmark."""
    from chatterbox.streaming import ChatterboxStreamer
    
    streamer = ChatterboxStreamer(model, chunk_tokens=5)
    
    # Warm-up
    print(f"  Warm-up run...")
    chunks = list(streamer.generate(text, language_id=language_id))
    print(f"  Warm-up complete: {len(chunks)} chunks")
    
    # Measured runs
    print(f"  Running {trials} trials...")
    results = []
    
    for i in range(trials):
        start = time.perf_counter()
        ttfa = None
        total_audio_ms = 0
        num_chunks = 0
        
        for chunk in streamer.generate(text, language_id=language_id):
            if ttfa is None:
                ttfa = (time.perf_counter() - start) * 1000
            total_audio_ms += chunk.audio_duration_ms
            num_chunks += 1
        
        total_time = (time.perf_counter() - start) * 1000
        rtf = total_time / total_audio_ms if total_audio_ms > 0 else 0
        
        results.append({
            'ttfa': ttfa,
            'total_time': total_time,
            'audio_ms': total_audio_ms,
            'rtf': rtf,
            'chunks': num_chunks,
        })
        
        print(f"    Trial {i+1}: TTFA={ttfa:.0f}ms, RTF={rtf:.2f}x, Chunks={num_chunks}")
    
    avg_ttfa = sum(r['ttfa'] for r in results) / len(results)
    avg_rtf = sum(r['rtf'] for r in results) / len(results)
    best_ttfa = min(r['ttfa'] for r in results)
    
    print(f"\n  {label} Results:")
    print(f"    Avg TTFA: {avg_ttfa:.0f}ms")
    print(f"    Best TTFA: {best_ttfa:.0f}ms")
    print(f"    Avg RTF: {avg_rtf:.2f}x")
    
    return {
        'avg_ttfa': avg_ttfa,
        'best_ttfa': best_ttfa,
        'avg_rtf': avg_rtf,
        'trials': results,
    }


def run_batch_test(model, text, language_id, trials, label):
    """Run batch (non-streaming) benchmark."""
    
    # Warm-up
    print(f"  Warm-up run...")
    _ = model.generate(text, language_id=language_id)
    print(f"  Warm-up complete")
    
    # Measured runs
    print(f"  Running {trials} trials...")
    results = []
    
    for i in range(trials):
        start = time.perf_counter()
        wav = model.generate(text, language_id=language_id)
        latency = (time.perf_counter() - start) * 1000
        
        audio_samples = wav.shape[-1]
        audio_ms = audio_samples / 22050 * 1000
        rtf = latency / audio_ms if audio_ms > 0 else 0
        
        results.append({
            'latency': latency,
            'audio_ms': audio_ms,
            'rtf': rtf,
        })
        
        print(f"    Trial {i+1}: Latency={latency:.0f}ms, Audio={audio_ms:.0f}ms, RTF={rtf:.2f}x")
    
    avg_latency = sum(r['latency'] for r in results) / len(results)
    best_latency = min(r['latency'] for r in results)
    avg_rtf = sum(r['rtf'] for r in results) / len(results)
    
    print(f"\n  {label} Results:")
    print(f"    Avg Latency: {avg_latency:.0f}ms")
    print(f"    Best Latency: {best_latency:.0f}ms")
    print(f"    Avg RTF: {avg_rtf:.2f}x")
    
    return {
        'avg_latency': avg_latency,
        'best_latency': best_latency,
        'avg_rtf': avg_rtf,
        'trials': results,
    }


def main():
    parser = argparse.ArgumentParser(description="Test MeanFlow hybrid setup")
    parser.add_argument(
        "--text", "-t",
        default="Hello, how are you today?",
        help="Text to synthesize"
    )
    parser.add_argument(
        "--lang", "-l",
        default="en",
        help="Language ID (default: en for proof-of-concept)"
    )
    parser.add_argument(
        "--arabic", "-a",
        action="store_true",
        help="Test with Arabic text"
    )
    parser.add_argument(
        "--trials", "-n",
        type=int,
        default=3,
        help="Number of trials"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use batch mode instead of streaming"
    )
    
    args = parser.parse_args()
    
    if args.arabic:
        args.text = "مرحباً، كيف حالك اليوم؟"
        args.lang = "ar"
    
    test_meanflow_hybrid(
        text=args.text,
        language_id=args.lang,
        trials=args.trials,
        streaming=not args.batch,
    )


if __name__ == "__main__":
    main()
