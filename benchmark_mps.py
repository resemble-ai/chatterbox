#!/usr/bin/env python3
"""
MPS Performance Benchmark for Chatterbox TTS
=============================================

Benchmarks text-to-speech generation on Apple Silicon (M4) comparing:
- MPS vs CPU performance
- Short vs Long text generation
- Memory usage patterns

Designed for MacBook Pro M4 32GB.
"""

import torch
import torchaudio as ta
import psutil
import os
import time
import gc
import statistics
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


# ============================================================================
# Benchmark Configuration
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    # Number of warmup runs (not counted in timing)
    warmup_runs: int = 1
    # Number of timed runs per test
    benchmark_runs: int = 3
    # Reference audio for voice cloning (optional)
    audio_prompt_path: Optional[str] = None
    # Output directory for generated audio
    output_dir: str = "benchmark_output"
    # Save generated audio files
    save_audio: bool = True
    # Test configurations
    test_devices: List[str] = field(default_factory=lambda: ["mps", "cpu"])
    # Model to benchmark: "standard" or "multilingual"
    model_type: str = "standard"


# ============================================================================
# Test Texts
# ============================================================================

SHORT_TEXTS = [
    "Hello, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "Welcome to the future of voice synthesis technology.",
]

MEDIUM_TEXTS = [
    """Machine learning is transforming how we interact with computers. 
    Natural language processing enables machines to understand and generate human speech. 
    This technology has applications in virtual assistants, accessibility tools, and entertainment.""",
    
    """The weather forecast for tomorrow shows partly cloudy skies with a high of seventy-two degrees. 
    There's a thirty percent chance of afternoon showers. 
    Winds will be light from the southwest at five to ten miles per hour.""",
]

LONG_TEXTS = [
    """Artificial intelligence has made remarkable strides in recent years, particularly in the field 
    of natural language processing. Text-to-speech systems have evolved from robotic-sounding voices 
    to remarkably natural human-like speech. These advances are powered by deep learning models that 
    learn from vast amounts of human speech data. The ability to generate natural-sounding speech has 
    numerous applications, from making content accessible to visually impaired users, to creating 
    audiobooks, to powering virtual assistants that feel more human. As the technology continues to 
    improve, we can expect even more lifelike and expressive synthetic voices in the near future.""",
    
    """The history of computing is a fascinating journey from mechanical calculators to the powerful 
    machines we have today. In the early twentieth century, pioneers like Alan Turing laid the 
    theoretical groundwork for modern computers. The first electronic computers were massive machines 
    that filled entire rooms and required constant maintenance. Over the decades, transistors replaced 
    vacuum tubes, and integrated circuits made computers smaller and more powerful. The personal 
    computer revolution of the nineteen eighties brought computing to homes and offices worldwide. 
    Today, we carry computers in our pockets that are millions of times more powerful than those 
    early machines. The pace of innovation shows no signs of slowing down.""",
]

# Extra long text for stress testing
EXTRA_LONG_TEXT = """
The development of artificial intelligence represents one of the most significant technological 
achievements in human history. From its theoretical foundations in the mid-twentieth century to 
today's sophisticated deep learning systems, AI has transformed virtually every aspect of modern life.

Early AI research focused on symbolic approaches, attempting to encode human knowledge into rule-based 
systems. While these systems showed promise in narrow domains like chess and mathematical theorem 
proving, they struggled with the complexity and ambiguity of real-world problems. The AI winters of 
the nineteen seventies and eighties saw funding and interest decline as the limitations of these 
approaches became apparent.

The resurgence of AI in the twenty-first century came from a different direction entirely. Machine 
learning, particularly deep learning with neural networks, proved remarkably effective at learning 
patterns from data. Breakthroughs in image recognition, natural language processing, and game playing 
captured the public imagination and attracted massive investment.

Today's AI systems can engage in natural conversations, generate creative content, drive vehicles, 
diagnose diseases, and perform countless other tasks that once seemed the exclusive domain of human 
intelligence. Yet significant challenges remain. Questions of bias, transparency, and safety require 
careful consideration as these systems become more integrated into critical applications.

The future of AI promises even more remarkable developments. Advances in multimodal learning, 
reasoning capabilities, and efficiency continue to expand what's possible. As we navigate this 
transformative technology, thoughtful development and deployment will be essential to ensuring 
that AI benefits humanity as a whole.
"""


# ============================================================================
# Utility Functions
# ============================================================================

def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def get_gpu_memory_mb(device: str) -> Optional[float]:
    """Get GPU memory usage if available."""
    if device == "mps" and torch.backends.mps.is_available():
        # MPS doesn't have direct memory query, but we can get allocation info
        try:
            return torch.mps.current_allocated_memory() / 1024 / 1024
        except AttributeError:
            return None
    elif device == "cuda" and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return None


def clear_memory(device: str):
    """Clear GPU memory and run garbage collection."""
    # Multiple rounds of GC for thorough cleanup
    for _ in range(3):
        gc.collect()
    
    if device == "mps" and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            torch.mps.synchronize()
        except Exception:
            pass
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Final GC pass
    gc.collect()


def word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def char_count(text: str) -> int:
    """Count characters in text."""
    return len(text)


# ============================================================================
# Benchmark Result Classes
# ============================================================================

@dataclass
class SingleRunResult:
    """Result from a single generation run."""
    device: str
    text_length_words: int
    text_length_chars: int
    generation_time_seconds: float
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    gpu_memory_mb: Optional[float]
    audio_duration_seconds: float
    realtime_factor: float  # audio_duration / generation_time


@dataclass
class BenchmarkResult:
    """Aggregated result from multiple benchmark runs."""
    device: str
    text_category: str  # "short", "medium", "long", "extra_long"
    text_preview: str
    text_length_words: int
    text_length_chars: int
    num_runs: int
    
    # Timing statistics (in seconds)
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    
    # Memory statistics
    mean_memory_delta_mb: float
    peak_memory_mb: float
    
    # Performance metrics
    mean_realtime_factor: float
    chars_per_second: float
    words_per_second: float
    
    # Audio info
    audio_duration_seconds: float


# ============================================================================
# Benchmark Runner
# ============================================================================

class ChatterboxBenchmark:
    """Benchmark runner for Chatterbox TTS."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.model = None
        self.current_device = None
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
    def load_model(self, device: str):
        """Load the TTS model on specified device."""
        print(f"\n{'='*60}")
        print(f"Loading model on {device.upper()}...")
        print(f"{'='*60}")
        
        # Clear any existing model
        if self.model is not None:
            del self.model
            self.model = None
            clear_memory(self.current_device or "cpu")
        
        # Check device availability
        if device == "mps":
            if not torch.backends.mps.is_available():
                print("⚠️  MPS not available, falling back to CPU")
                device = "cpu"
            else:
                print("✓ MPS (Apple Silicon) is available")
        elif device == "cuda":
            if not torch.cuda.is_available():
                print("⚠️  CUDA not available, falling back to CPU")
                device = "cpu"
        
        self.current_device = device
        mem_before = get_memory_mb()
        load_start = time.time()
        
        if self.config.model_type == "multilingual":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            self.model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        else:
            from chatterbox.tts import ChatterboxTTS
            self.model = ChatterboxTTS.from_pretrained(device=device)
        
        load_time = time.time() - load_start
        mem_after = get_memory_mb()
        
        print(f"✓ Model loaded in {load_time:.2f}s")
        print(f"  Memory: {mem_before:.1f} → {mem_after:.1f} MB (+{mem_after - mem_before:.1f} MB)")
        
        # Prepare conditionals if audio prompt provided
        if self.config.audio_prompt_path:
            print(f"  Loading voice from: {self.config.audio_prompt_path}")
            self.model.prepare_conditionals(self.config.audio_prompt_path)
            print("  ✓ Voice conditionals prepared")
    
    def run_single_generation(self, text: str, run_id: int = 0) -> SingleRunResult:
        """Run a single TTS generation and collect metrics."""
        clear_memory(self.current_device)
        
        mem_before = get_memory_mb()
        gpu_mem_before = get_gpu_memory_mb(self.current_device)
        
        start_time = time.time()
        
        # Generate audio
        if self.config.model_type == "multilingual":
            wav = self.model.generate(
                text,
                language_id="en",
                audio_prompt_path=self.config.audio_prompt_path if run_id == 0 else None,
                exaggeration=0.5,
                cfg_weight=0.5,
            )
        else:
            wav = self.model.generate(
                text,
                audio_prompt_path=self.config.audio_prompt_path if run_id == 0 else None,
                exaggeration=0.5,
                cfg_weight=0.5,
            )
        
        generation_time = time.time() - start_time
        
        mem_after = get_memory_mb()
        gpu_mem_after = get_gpu_memory_mb(self.current_device)
        
        # Calculate audio duration
        sample_rate = self.model.sr
        audio_samples = wav.shape[-1]
        audio_duration = audio_samples / sample_rate
        
        # Calculate real-time factor
        realtime_factor = audio_duration / generation_time if generation_time > 0 else 0
        
        result = SingleRunResult(
            device=self.current_device,
            text_length_words=word_count(text),
            text_length_chars=char_count(text),
            generation_time_seconds=generation_time,
            memory_before_mb=mem_before,
            memory_after_mb=mem_after,
            memory_delta_mb=mem_after - mem_before,
            gpu_memory_mb=gpu_mem_after,
            audio_duration_seconds=audio_duration,
            realtime_factor=realtime_factor,
        )
        
        # CRITICAL: Explicitly delete tensor and clear memory to prevent leaks
        del wav
        clear_memory(self.current_device)
        
        return result
    
    def benchmark_text(self, text: str, category: str) -> BenchmarkResult:
        """Run complete benchmark on a single text."""
        text_words = word_count(text)
        text_chars = char_count(text)
        preview = text[:50] + "..." if len(text) > 50 else text
        
        print(f"\n  Testing: {preview}")
        print(f"  Length: {text_words} words, {text_chars} chars")
        
        # Warmup runs
        print(f"  Warmup runs: ", end="", flush=True)
        for i in range(self.config.warmup_runs):
            self.run_single_generation(text, run_id=i)
            print(".", end="", flush=True)
        print(" done")
        
        # Benchmark runs
        print(f"  Benchmark runs: ", end="", flush=True)
        run_results: List[SingleRunResult] = []
        for i in range(self.config.benchmark_runs):
            result = self.run_single_generation(text, run_id=i + self.config.warmup_runs)
            run_results.append(result)
            print(".", end="", flush=True)
        print(" done")
        
        # Aggregate results
        times = [r.generation_time_seconds for r in run_results]
        memory_deltas = [r.memory_delta_mb for r in run_results]
        peak_memory = max(r.memory_after_mb for r in run_results)
        realtime_factors = [r.realtime_factor for r in run_results]
        audio_duration = run_results[0].audio_duration_seconds
        
        mean_time = statistics.mean(times)
        
        result = BenchmarkResult(
            device=self.current_device,
            text_category=category,
            text_preview=preview,
            text_length_words=text_words,
            text_length_chars=text_chars,
            num_runs=self.config.benchmark_runs,
            mean_time=mean_time,
            std_time=statistics.stdev(times) if len(times) > 1 else 0,
            min_time=min(times),
            max_time=max(times),
            mean_memory_delta_mb=statistics.mean(memory_deltas),
            peak_memory_mb=peak_memory,
            mean_realtime_factor=statistics.mean(realtime_factors),
            chars_per_second=text_chars / mean_time if mean_time > 0 else 0,
            words_per_second=text_words / mean_time if mean_time > 0 else 0,
            audio_duration_seconds=audio_duration,
        )
        
        print(f"  → Time: {result.mean_time:.2f}s ± {result.std_time:.2f}s")
        print(f"  → Real-time factor: {result.mean_realtime_factor:.2f}x")
        print(f"  → Throughput: {result.words_per_second:.1f} words/s, {result.chars_per_second:.1f} chars/s")
        
        # Force aggressive cleanup between benchmarks
        clear_memory(self.current_device)
        
        return result
    
    def run_benchmarks(self):
        """Run all benchmarks."""
        print("\n" + "="*70)
        print("CHATTERBOX TTS BENCHMARK")
        print("="*70)
        print(f"Platform: macOS with Apple Silicon")
        print(f"MPS Available: {torch.backends.mps.is_available()}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Model Type: {self.config.model_type}")
        print(f"Warmup Runs: {self.config.warmup_runs}")
        print(f"Benchmark Runs: {self.config.benchmark_runs}")
        print(f"Devices to Test: {', '.join(self.config.test_devices)}")
        print("="*70)
        
        all_tests = [
            ("short", SHORT_TEXTS[0]),
            ("medium", MEDIUM_TEXTS[0]),
            ("long", LONG_TEXTS[0]),
            ("extra_long", EXTRA_LONG_TEXT),
        ]
        
        for device in self.config.test_devices:
            self.load_model(device)
            
            print(f"\n{'─'*60}")
            print(f"BENCHMARKING ON {device.upper()}")
            print(f"{'─'*60}")
            
            for category, text in all_tests:
                result = self.benchmark_text(text, category)
                self.results.append(result)
                
                # Save audio if requested
                if self.config.save_audio:
                    # Generate one final time for saving
                    if self.config.model_type == "multilingual":
                        wav = self.model.generate(
                            text,
                            language_id="en",
                            exaggeration=0.5,
                            cfg_weight=0.5,
                        )
                    else:
                        wav = self.model.generate(
                            text,
                            exaggeration=0.5,
                            cfg_weight=0.5,
                        )
                    
                    output_path = Path(self.config.output_dir) / f"{device}_{category}.wav"
                    ta.save(str(output_path), wav, self.model.sr)
                    print(f"  → Saved: {output_path}")
    
    def print_summary(self):
        """Print benchmark summary table."""
        print("\n" + "="*90)
        print("BENCHMARK SUMMARY")
        print("="*90)
        
        # Group results by device
        devices = list(set(r.device for r in self.results))
        categories = ["short", "medium", "long", "extra_long"]
        
        # Header
        print(f"\n{'Category':<12} {'Words':<8} ", end="")
        for device in devices:
            print(f"│ {device.upper():<25} ", end="")
        print()
        
        print(f"{'─'*12} {'─'*8} ", end="")
        for device in devices:
            print(f"│ {'─'*25} ", end="")
        print()
        
        # Data rows
        for category in categories:
            cat_results = [r for r in self.results if r.text_category == category]
            if not cat_results:
                continue
            
            words = cat_results[0].text_length_words
            print(f"{category:<12} {words:<8} ", end="")
            
            for device in devices:
                dev_result = next((r for r in cat_results if r.device == device), None)
                if dev_result:
                    time_str = f"{dev_result.mean_time:.2f}s"
                    rtf_str = f"{dev_result.mean_realtime_factor:.2f}x"
                    print(f"│ {time_str:<10} {rtf_str:<13} ", end="")
                else:
                    print(f"│ {'N/A':<25} ", end="")
            print()
        
        # Speedup comparison if multiple devices
        if len(devices) > 1 and "cpu" in devices and "mps" in devices:
            print(f"\n{'─'*90}")
            print("MPS vs CPU Speedup:")
            for category in categories:
                cat_results = [r for r in self.results if r.text_category == category]
                cpu_result = next((r for r in cat_results if r.device == "cpu"), None)
                mps_result = next((r for r in cat_results if r.device == "mps"), None)
                
                if cpu_result and mps_result and cpu_result.mean_time > 0:
                    speedup = cpu_result.mean_time / mps_result.mean_time
                    faster = "MPS" if speedup > 1 else "CPU"
                    ratio = speedup if speedup > 1 else 1/speedup
                    print(f"  {category:<12}: {faster} is {ratio:.2f}x faster")
        
        print(f"\n{'='*90}")
        
    def export_results(self, filepath: str = None):
        """Export results to JSON."""
        import json
        
        if filepath is None:
            filepath = Path(self.config.output_dir) / "benchmark_results.json"
        
        data = {
            "config": {
                "warmup_runs": self.config.warmup_runs,
                "benchmark_runs": self.config.benchmark_runs,
                "model_type": self.config.model_type,
                "test_devices": self.config.test_devices,
            },
            "system": {
                "mps_available": torch.backends.mps.is_available(),
                "pytorch_version": torch.__version__,
                "total_ram_gb": psutil.virtual_memory().total / (1024**3),
            },
            "results": [
                {
                    "device": r.device,
                    "text_category": r.text_category,
                    "text_length_words": r.text_length_words,
                    "text_length_chars": r.text_length_chars,
                    "mean_time_seconds": r.mean_time,
                    "std_time_seconds": r.std_time,
                    "min_time_seconds": r.min_time,
                    "max_time_seconds": r.max_time,
                    "mean_realtime_factor": r.mean_realtime_factor,
                    "chars_per_second": r.chars_per_second,
                    "words_per_second": r.words_per_second,
                    "audio_duration_seconds": r.audio_duration_seconds,
                    "peak_memory_mb": r.peak_memory_mb,
                }
                for r in self.results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults exported to: {filepath}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chatterbox TTS MPS Benchmark")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--devices", nargs="+", default=["mps", "cpu"], 
                       help="Devices to benchmark (mps, cpu)")
    parser.add_argument("--model", choices=["standard", "multilingual"], 
                       default="standard", help="Model type to benchmark")
    parser.add_argument("--audio-prompt", type=str, default=None,
                       help="Path to reference audio for voice cloning")
    parser.add_argument("--output-dir", type=str, default="benchmark_output",
                       help="Output directory for results")
    parser.add_argument("--no-save-audio", action="store_true",
                       help="Don't save generated audio files")
    parser.add_argument("--mps-only", action="store_true",
                       help="Only benchmark MPS (skip CPU)")
    parser.add_argument("--cpu-only", action="store_true",
                       help="Only benchmark CPU (skip MPS)")
    
    args = parser.parse_args()
    
    # Handle device selection
    if args.mps_only:
        devices = ["mps"]
    elif args.cpu_only:
        devices = ["cpu"]
    else:
        devices = args.devices
    
    config = BenchmarkConfig(
        warmup_runs=args.warmup,
        benchmark_runs=args.runs,
        test_devices=devices,
        model_type=args.model,
        audio_prompt_path=args.audio_prompt,
        output_dir=args.output_dir,
        save_audio=not args.no_save_audio,
    )
    
    benchmark = ChatterboxBenchmark(config)
    
    try:
        benchmark.run_benchmarks()
        benchmark.print_summary()
        benchmark.export_results()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        if benchmark.results:
            benchmark.print_summary()
    except Exception as e:
        print(f"\n\nBenchmark failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
