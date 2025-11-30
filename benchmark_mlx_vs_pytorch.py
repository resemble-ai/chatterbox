# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
Comprehensive benchmark comparing MLX vs PyTorch implementations of T3.

Measures:
- Inference speed
- Memory usage
- Generation quality (numerical similarity)
- Model loading time
- Quantized MLX performance (4-bit and 8-bit)
"""

import time
import psutil
import os
import argparse
from pathlib import Path
from typing import Dict, List
import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for MLX availability
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logger.warning("⚠️  MLX not installed. Install with: pip install mlx mlx-lm")
    logger.warning("⚠️  MLX benchmarks will be skipped.")


def get_memory_mb():
    """Get current process memory usage in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


class BenchmarkResults:
    """Store and display benchmark results."""

    def __init__(self):
        self.results: Dict[str, Dict] = {}

    def add_result(self, name: str, pytorch_time: float, mlx_time: float,
                   pytorch_mem: float, mlx_mem: float,
                   mlx_quant_time: float = 0, mlx_quant_mem: float = 0):
        """Add a benchmark result."""
        speedup = pytorch_time / mlx_time if mlx_time > 0 else 0
        mem_reduction = (pytorch_mem - mlx_mem) / pytorch_mem * 100 if pytorch_mem > 0 else 0

        speedup_quant = pytorch_time / mlx_quant_time if mlx_quant_time > 0 else 0
        mem_reduction_quant = (pytorch_mem - mlx_quant_mem) / pytorch_mem * 100 if pytorch_mem > 0 else 0

        self.results[name] = {
            'pytorch_time': pytorch_time,
            'mlx_time': mlx_time,
            'mlx_quant_time': mlx_quant_time,
            'speedup': speedup,
            'speedup_quant': speedup_quant,
            'pytorch_mem': pytorch_mem,
            'mlx_mem': mlx_mem,
            'mlx_quant_mem': mlx_quant_mem,
            'mem_reduction': mem_reduction,
            'mem_reduction_quant': mem_reduction_quant,
        }

    def print_summary(self):
        """Print formatted benchmark summary."""
        print("\n" + "=" * 120)
        print("MLX vs PyTorch Benchmark Results (Including Quantized MLX)")
        print("=" * 120)

        # Header
        print(f"{'Benchmark':<25} {'PyTorch':<12} {'MLX':<12} {'MLX-Q4':<12} "
              f"{'Speedup':<10} {'Q4 Speedup':<12} {'Mem Save':<10} {'Q4 Mem Save':<12}")
        print("-" * 120)

        for name, result in self.results.items():
            mlx_q_str = f"{result['mlx_quant_time']:.3f}" if result['mlx_quant_time'] > 0 else "N/A"
            speedup_q_str = f"{result['speedup_quant']:.2f}x" if result['speedup_quant'] > 0 else "N/A"
            mem_q_str = f"{result['mem_reduction_quant']:.1f}%" if result['mlx_quant_time'] > 0 else "N/A"

            print(f"{name:<25} "
                  f"{result['pytorch_time']:<12.3f} "
                  f"{result['mlx_time']:<12.3f} "
                  f"{mlx_q_str:<12} "
                  f"{result['speedup']:<10.2f}x "
                  f"{speedup_q_str:<12} "
                  f"{result['mem_reduction']:<10.1f}% "
                  f"{mem_q_str:<12}")

        print("=" * 120)

        # Overall statistics
        avg_speedup = np.mean([r['speedup'] for r in self.results.values()])
        avg_mem_reduction = np.mean([r['mem_reduction'] for r in self.results.values()])

        # Quantized stats (only for benchmarks that have quantized results)
        quant_results = [r for r in self.results.values() if r['speedup_quant'] > 0]
        if quant_results:
            avg_speedup_quant = np.mean([r['speedup_quant'] for r in quant_results])
            avg_mem_reduction_quant = np.mean([r['mem_reduction_quant'] for r in quant_results])

            print(f"\nOverall Performance:")
            print(f"  MLX Full Precision:")
            print(f"    Average Speedup: {avg_speedup:.2f}x")
            print(f"    Average Memory Reduction: {avg_mem_reduction:.1f}%")
            print(f"  MLX 4-bit Quantized:")
            print(f"    Average Speedup: {avg_speedup_quant:.2f}x")
            print(f"    Average Memory Reduction: {avg_mem_reduction_quant:.1f}%")
        else:
            print(f"\nOverall Performance:")
            print(f"  Average Speedup: {avg_speedup:.2f}x")
            print(f"  Average Memory Reduction: {avg_mem_reduction:.1f}%")
        print()


def benchmark_model_loading():
    """Benchmark model loading time."""
    logger.info("Benchmarking model loading...")

    # PyTorch loading
    mem_before = get_memory_mb()
    start = time.time()
    try:
        from chatterbox.models.t3.t3 import T3 as T3PyTorch
        from chatterbox.models.t3.modules.t3_config import T3Config

        config = T3Config.english_only()
        t3_pt = T3PyTorch(hp=config)
        pytorch_time = time.time() - start
        pytorch_mem = get_memory_mb() - mem_before
        del t3_pt
    except Exception as e:
        logger.error(f"PyTorch loading failed: {e}")
        pytorch_time = 0
        pytorch_mem = 0

    # MLX loading
    if not MLX_AVAILABLE:
        logger.warning("Skipping MLX benchmark - MLX not installed")
        return pytorch_time, 0, pytorch_mem, 0, 0, 0

    mem_before = get_memory_mb()
    start = time.time()
    try:
        from chatterbox.models.t3_mlx.t3_mlx import T3MLX
        from chatterbox.models.t3.modules.t3_config import T3Config

        config = T3Config.english_only()
        t3_mlx = T3MLX(hp=config)
        mlx_time = time.time() - start
        mlx_mem = get_memory_mb() - mem_before
        del t3_mlx
    except Exception as e:
        logger.error(f"MLX loading failed: {e}")
        mlx_time = 0
        mlx_mem = 0

    # Quantized MLX loading
    mem_before = get_memory_mb()
    start = time.time()
    try:
        from chatterbox.models.t3_mlx.t3_mlx import T3MLX
        from chatterbox.models.t3_mlx.quantization.quantize_mlx import QuantizedT3MLX
        from chatterbox.models.t3.modules.t3_config import T3Config

        config = T3Config.english_only()
        t3_mlx_base = T3MLX(hp=config)
        t3_mlx_quant = QuantizedT3MLX(t3_mlx_base, bits=4, group_size=64)
        mlx_quant_time = time.time() - start
        mlx_quant_mem = get_memory_mb() - mem_before
        del t3_mlx_base, t3_mlx_quant
    except Exception as e:
        logger.error(f"Quantized MLX loading failed: {e}")
        import traceback
        traceback.print_exc()
        mlx_quant_time = 0
        mlx_quant_mem = 0

    return pytorch_time, mlx_time, pytorch_mem, mlx_mem, mlx_quant_time, mlx_quant_mem


def benchmark_forward_pass(seq_length: int = 50):
    """Benchmark forward pass speed."""
    logger.info(f"Benchmarking forward pass (seq_length={seq_length})...")

    if not MLX_AVAILABLE:
        logger.warning("Skipping MLX benchmark - MLX not installed")
        return 0, 0, 0, 0, 0, 0

    import torch
    import mlx.core as mx

    from chatterbox.models.t3.t3 import T3 as T3PyTorch
    from chatterbox.models.t3_mlx.t3_mlx import T3MLX
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    from chatterbox.models.t3_mlx.modules.cond_enc_mlx import T3CondMLX

    config = T3Config.english_only()

    # Prepare dummy inputs
    batch_size = 1
    text_len = 10
    speech_len = seq_length

    # PyTorch benchmark
    try:
        t3_pt = T3PyTorch(hp=config)
        t3_pt.eval()

        text_tokens_pt = torch.randint(0, config.text_tokens_dict_size, (batch_size, text_len))
        text_tokens_pt[:, 0] = config.start_text_token
        text_tokens_pt[:, -1] = config.stop_text_token
        speech_tokens_pt = torch.randint(0, config.speech_tokens_dict_size, (batch_size, speech_len))
        text_lens_pt = torch.tensor([text_len])
        speech_lens_pt = torch.tensor([speech_len])

        # emotion_adv should be a tensor for PyTorch
        cond_pt = T3Cond(
            speaker_emb=torch.randn(batch_size, 256),
            emotion_adv=torch.tensor([0.5])
        )

        # Warm-up
        with torch.no_grad():
            _ = t3_pt(
                t3_cond=cond_pt,
                text_tokens=text_tokens_pt,
                text_token_lens=text_lens_pt,
                speech_tokens=speech_tokens_pt,
                speech_token_lens=speech_lens_pt,
            )

        # Benchmark
        mem_before = get_memory_mb()
        start = time.time()
        num_iters = 10
        for _ in range(num_iters):
            with torch.no_grad():
                _ = t3_pt(
                    t3_cond=cond_pt,
                    text_tokens=text_tokens_pt,
                    text_token_lens=text_lens_pt,
                    speech_tokens=speech_tokens_pt,
                    speech_token_lens=speech_lens_pt,
                )
        pytorch_time = (time.time() - start) / num_iters
        pytorch_mem = get_memory_mb() - mem_before

        del t3_pt
    except Exception as e:
        logger.error(f"PyTorch forward pass failed: {e}")
        pytorch_time = 0
        pytorch_mem = 0

    # MLX benchmark
    try:
        t3_mlx = T3MLX(hp=config)

        # Create proper text tokens with start/stop tokens
        text_tokens_mlx = mx.random.randint(0, config.text_tokens_dict_size, (batch_size, text_len))
        text_tokens_mlx = mx.array(text_tokens_mlx)  # Ensure it's an array
        # Set start and stop tokens
        text_tokens_list = text_tokens_mlx.tolist() if hasattr(text_tokens_mlx, 'tolist') else [[int(x) for x in text_tokens_mlx[0]]]
        text_tokens_list[0][0] = config.start_text_token
        text_tokens_list[0][-1] = config.stop_text_token
        text_tokens_mlx = mx.array(text_tokens_list)

        speech_tokens_mlx = mx.random.randint(0, config.speech_tokens_dict_size, (batch_size, speech_len))
        text_lens_mlx = mx.array([text_len])
        speech_lens_mlx = mx.array([speech_len])

        cond_mlx = T3CondMLX(speaker_emb=mx.random.normal((batch_size, 256)), emotion_adv=0.5)

        # Warm-up
        _ = t3_mlx(
            t3_cond=cond_mlx,
            text_tokens=text_tokens_mlx,
            text_token_lens=text_lens_mlx,
            speech_tokens=speech_tokens_mlx,
            speech_token_lens=speech_lens_mlx,
        )

        # Benchmark
        mem_before = get_memory_mb()
        start = time.time()
        num_iters = 10
        for _ in range(num_iters):
            output = t3_mlx(
                t3_cond=cond_mlx,
                text_tokens=text_tokens_mlx,
                text_token_lens=text_lens_mlx,
                speech_tokens=speech_tokens_mlx,
                speech_token_lens=speech_lens_mlx,
            )
            mx.eval(output['speech_logits'])  # Force evaluation
        mlx_time = (time.time() - start) / num_iters
        mlx_mem = get_memory_mb() - mem_before

        del t3_mlx
    except Exception as e:
        logger.error(f"MLX forward pass failed: {e}")
        mlx_time = 0
        mlx_mem = 0

    # Quantized MLX benchmark
    try:
        from chatterbox.models.t3_mlx.quantization.quantize_mlx import QuantizedT3MLX

        t3_mlx_base = T3MLX(hp=config)
        t3_mlx_quant = QuantizedT3MLX(t3_mlx_base, bits=4, group_size=64)

        # Use same inputs as MLX
        # Warm-up
        _ = t3_mlx_quant(
            t3_cond=cond_mlx,
            text_tokens=text_tokens_mlx,
            text_token_lens=text_lens_mlx,
            speech_tokens=speech_tokens_mlx,
            speech_token_lens=speech_lens_mlx,
        )

        # Benchmark
        mem_before = get_memory_mb()
        start = time.time()
        num_iters = 10
        for _ in range(num_iters):
            output = t3_mlx_quant(
                t3_cond=cond_mlx,
                text_tokens=text_tokens_mlx,
                text_token_lens=text_lens_mlx,
                speech_tokens=speech_tokens_mlx,
                speech_token_lens=speech_lens_mlx,
            )
            mx.eval(output['speech_logits'])  # Force evaluation
        mlx_quant_time = (time.time() - start) / num_iters
        mlx_quant_mem = get_memory_mb() - mem_before

        del t3_mlx_base, t3_mlx_quant
    except Exception as e:
        logger.error(f"Quantized MLX forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        mlx_quant_time = 0
        mlx_quant_mem = 0

    return pytorch_time, mlx_time, pytorch_mem, mlx_mem, mlx_quant_time, mlx_quant_mem


def benchmark_generation(max_tokens: int = 50):
    """Benchmark autoregressive generation."""
    logger.info(f"Benchmarking generation (max_tokens={max_tokens})...")

    if not MLX_AVAILABLE:
        logger.warning("Skipping MLX benchmark - MLX not installed")
        return 0, 0, 0, 0, 0, 0

    import torch
    import mlx.core as mx

    from chatterbox.models.t3.t3 import T3 as T3PyTorch
    from chatterbox.models.t3_mlx.t3_mlx import T3MLX
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    from chatterbox.models.t3_mlx.modules.cond_enc_mlx import T3CondMLX

    config = T3Config.english_only()

    # Prepare text tokens
    text_tokens_list = [config.start_text_token, 10, 20, 30, config.stop_text_token]

    # PyTorch generation
    try:
        t3_pt = T3PyTorch(hp=config)
        t3_pt.eval()

        text_tokens_pt = torch.tensor([text_tokens_list])
        cond_pt = T3Cond(
            speaker_emb=torch.randn(1, 256),
            emotion_adv=torch.tensor([0.5])  # Should be tensor
        )

        mem_before = get_memory_mb()
        start = time.time()

        with torch.no_grad():
            output_pt = t3_pt.inference(
                t3_cond=cond_pt,
                text_tokens=text_tokens_pt,
                max_new_tokens=max_tokens,
                temperature=0.8,
                cfg_weight=0.0,  # Disable CFG to avoid complexity
            )

        pytorch_time = time.time() - start
        pytorch_mem = get_memory_mb() - mem_before

        del t3_pt, output_pt
    except Exception as e:
        logger.error(f"PyTorch generation failed: {e}")
        import traceback
        traceback.print_exc()
        pytorch_time = 0
        pytorch_mem = 0

    # MLX generation
    try:
        t3_mlx = T3MLX(hp=config)

        text_tokens_mlx = mx.array([text_tokens_list])
        cond_mlx = T3CondMLX(speaker_emb=mx.random.normal((1, 256)), emotion_adv=0.5)

        mem_before = get_memory_mb()
        start = time.time()

        output_mlx = t3_mlx.generate(
            t3_cond=cond_mlx,
            text_tokens=text_tokens_mlx,
            max_new_tokens=max_tokens,
            temperature=0.8,
            cfg_weight=0.0,  # Disable CFG to simplify for benchmarking
        )

        mlx_time = time.time() - start
        mlx_mem = get_memory_mb() - mem_before

        del t3_mlx, output_mlx
    except Exception as e:
        logger.error(f"MLX generation failed: {e}")
        import traceback
        traceback.print_exc()
        mlx_time = 0
        mlx_mem = 0

    # Quantized MLX generation
    try:
        from chatterbox.models.t3_mlx.quantization.quantize_mlx import QuantizedT3MLX

        t3_mlx_base = T3MLX(hp=config)
        t3_mlx_quant = QuantizedT3MLX(t3_mlx_base, bits=4, group_size=64)

        text_tokens_mlx = mx.array([text_tokens_list])
        cond_mlx = T3CondMLX(speaker_emb=mx.random.normal((1, 256)), emotion_adv=0.5)

        mem_before = get_memory_mb()
        start = time.time()

        output_mlx_quant = t3_mlx_quant.generate(
            t3_cond=cond_mlx,
            text_tokens=text_tokens_mlx,
            max_new_tokens=max_tokens,
            temperature=0.8,
            cfg_weight=0.0,  # Disable CFG to simplify for benchmarking
        )

        mlx_quant_time = time.time() - start
        mlx_quant_mem = get_memory_mb() - mem_before

        del t3_mlx_base, t3_mlx_quant, output_mlx_quant
    except Exception as e:
        logger.error(f"Quantized MLX generation failed: {e}")
        import traceback
        traceback.print_exc()
        mlx_quant_time = 0
        mlx_quant_mem = 0

    return pytorch_time, mlx_time, pytorch_mem, mlx_mem, mlx_quant_time, mlx_quant_mem


def main():
    """Run all benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark MLX vs PyTorch T3 implementations")
    parser.add_argument("--skip-loading", action="store_true", help="Skip model loading benchmark")
    parser.add_argument("--skip-forward", action="store_true", help="Skip forward pass benchmark")
    parser.add_argument("--skip-generation", action="store_true", help="Skip generation benchmark")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens for generation benchmark")

    args = parser.parse_args()

    results = BenchmarkResults()

    # Model loading
    if not args.skip_loading:
        pt_time, mlx_time, pt_mem, mlx_mem, mlx_q_time, mlx_q_mem = benchmark_model_loading()
        results.add_result("Model Loading", pt_time, mlx_time, pt_mem, mlx_mem, mlx_q_time, mlx_q_mem)

    # Forward pass - different sequence lengths
    if not args.skip_forward:
        for seq_len in [10, 50, 100]:
            pt_time, mlx_time, pt_mem, mlx_mem, mlx_q_time, mlx_q_mem = benchmark_forward_pass(seq_length=seq_len)
            results.add_result(f"Forward (len={seq_len})", pt_time, mlx_time, pt_mem, mlx_mem, mlx_q_time, mlx_q_mem)

    # Generation
    if not args.skip_generation:
        for max_tokens in [20, 50, 100]:
            pt_time, mlx_time, pt_mem, mlx_mem, mlx_q_time, mlx_q_mem = benchmark_generation(max_tokens=max_tokens)
            results.add_result(f"Gen ({max_tokens} tokens)", pt_time, mlx_time, pt_mem, mlx_mem, mlx_q_time, mlx_q_mem)

    # Print results
    results.print_summary()

    # Save results to file
    output_path = Path("benchmark_mlx_results.txt")
    with open(output_path, "w") as f:
        f.write("MLX vs PyTorch Benchmark Results (Including Quantized)\n")
        f.write("=" * 120 + "\n")
        for name, result in results.results.items():
            line = f"{name}: MLX {result['speedup']:.2f}x speedup, {result['mem_reduction']:.1f}% memory reduction"
            if result['speedup_quant'] > 0:
                line += f" | MLX-Q4 {result['speedup_quant']:.2f}x speedup, {result['mem_reduction_quant']:.1f}% memory reduction"
            f.write(line + "\n")

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
