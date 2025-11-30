#!/usr/bin/env python3
"""
Multilingual TTS Benchmark for Chatterbox
==========================================

Benchmarks multilingual text-to-speech generation with voice cloning:
- Tests multiple languages with native text samples
- Supports reference audio for voice cloning
- Compares MPS (PyTorch), Hybrid MLX (T3 MLX + S3Gen PyTorch), MLX, and MLX Quantized (4-bit) performance
- Measures generation quality across languages
- Optional Whisper transcription validation for quality assessment

NOTE: Float16 KV cache optimization is enabled by default for PyTorch models,
providing 18-32% speed improvements with significant memory savings.
MLX models use native optimizations, MLX-Q4 uses 4-bit quantization.

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
from chatterbox.models.utils import get_memory_info


# ============================================================================
# Benchmark Configuration
# ============================================================================

@dataclass
class MultilingualBenchmarkConfig:
    """Configuration for multilingual benchmark runs."""
    # Number of warmup runs (not counted in timing)
    warmup_runs: int = 1
    # Number of timed runs per test
    benchmark_runs: int = 3
    # Reference audio for voice cloning (REQUIRED for multilingual)
    audio_prompt_path: Optional[str] = None
    # Output directory for generated audio
    output_dir: str = "benchmark_multilingual_output"
    # Save generated audio files
    save_audio: bool = True
    # Test configurations
    test_devices: List[str] = field(default_factory=lambda: ["mps", "cpu", "hybrid-mlx", "mlx", "mlx-q4"])
    # Languages to test (subset of supported languages)
    test_languages: List[str] = field(default_factory=lambda: ["en", "es", "fr", "de", "ja", "zh"])
    # Enable Whisper transcription validation
    validate_transcription: bool = False
    # Note: MLX backends will be skipped gracefully if not yet available


# ============================================================================
# Multilingual Test Texts
# ============================================================================

# Dictionary of test texts by language code
# Format: language_code -> (language_name, test_text)
MULTILINGUAL_TEXTS = {
    "en": (
        "English",
        "Hello, this is a test of multilingual speech synthesis. The technology can generate natural-sounding voices in many different languages.",
    ),
    "es": (
        "Spanish",
        "Hola, esta es una prueba de síntesis de voz multilingüe. La tecnología puede generar voces de sonido natural en muchos idiomas diferentes.",
    ),
    "fr": (
        "French",
        "Bonjour, ceci est un test de synthèse vocale multilingue. La technologie peut générer des voix naturelles dans de nombreuses langues différentes.",
    ),
    "de": (
        "German",
        "Hallo, dies ist ein Test der mehrsprachigen Sprachsynthese. Die Technologie kann natürlich klingende Stimmen in vielen verschiedenen Sprachen erzeugen.",
    ),
    "it": (
        "Italian",
        "Ciao, questo è un test di sintesi vocale multilingue. La tecnologia può generare voci dal suono naturale in molte lingue diverse.",
    ),
    "pt": (
        "Portuguese",
        "Olá, este é um teste de síntese de fala multilíngue. A tecnologia pode gerar vozes de som natural em muitos idiomas diferentes.",
    ),
    "ru": (
        "Russian",
        "Здравствуйте, это тест многоязычного синтеза речи. Технология может генерировать естественно звучащие голоса на многих различных языках.",
    ),
    "ja": (
        "Japanese",
        "こんにちは、これは多言語音声合成のテストです。この技術は、さまざまな言語で自然な音声を生成できます。",
    ),
    "zh": (
        "Chinese",
        "你好，这是多语言语音合成的测试。该技术可以生成许多不同语言的自然语音。",
    ),
    "ko": (
        "Korean",
        "안녕하세요, 이것은 다국어 음성 합성 테스트입니다. 이 기술은 다양한 언어로 자연스러운 음성을 생성할 수 있습니다.",
    ),
    "ar": (
        "Arabic",
        "مرحبا، هذا اختبار لتوليف الكلام متعدد اللغات. يمكن للتكنولوجيا توليد أصوات طبيعية في العديد من اللغات المختلفة.",
    ),
    "hi": (
        "Hindi",
        "नमस्ते, यह बहुभाषी भाषण संश्लेषण का परीक्षण है। प्रौद्योगिकी कई विभिन्न भाषाओं में प्राकृतिक-ध्वनि वाली आवाज़ें उत्पन्न कर सकती है।",
    ),
    "tr": (
        "Turkish",
        "Merhaba, bu çok dilli konuşma sentezi testidir. Teknoloji, birçok farklı dilde doğal sesli sesler üretebilir.",
    ),
    "pl": (
        "Polish",
        "Witaj, to jest test wielojęzycznej syntezy mowy. Technologia może generować naturalnie brzmiące głosy w wielu różnych językach.",
    ),
    "nl": (
        "Dutch",
        "Hallo, dit is een test van meertalige spraaksynthese. De technologie kan natuurlijk klinkende stemmen genereren in veel verschillende talen.",
    ),
    "sv": (
        "Swedish",
        "Hej, det här är ett test av flerspråkig talsyntes. Tekniken kan generera naturligt ljudande röster på många olika språk.",
    ),
    "da": (
        "Danish",
        "Hej, dette er en test af flersproget talesyntese. Teknologien kan generere naturligt lydende stemmer på mange forskellige sprog.",
    ),
    "no": (
        "Norwegian",
        "Hei, dette er en test av flerspråklig talesyntese. Teknologien kan generere naturlig lydende stemmer på mange forskjellige språk.",
    ),
    "fi": (
        "Finnish",
        "Hei, tämä on monikielisen puhesynteesin testi. Teknologia voi tuottaa luonnollisen kuuloisia ääniä monilla eri kielillä.",
    ),
    "el": (
        "Greek",
        "Γεια σας, αυτό είναι μια δοκιμή πολυγλωσσικής σύνθεσης ομιλίας. Η τεχνολογία μπορεί να δημιουργήσει φυσικές φωνές σε πολλές διαφορετικές γλώσσες.",
    ),
    "he": (
        "Hebrew",
        "שלום, זהו מבחן של סינתזת דיבור רב לשוני. הטכנולוגיה יכולה ליצור קולות טבעיים בשפות שונות רבות.",
    ),
    "ms": (
        "Malay",
        "Halo, ini adalah ujian sintesis pertuturan berbilang bahasa. Teknologi ini boleh menghasilkan suara yang berbunyi semula jadi dalam banyak bahasa yang berbeza.",
    ),
    "sw": (
        "Swahili",
        "Habari, huu ni mtihani wa usanisi wa hotuba ya lugha nyingi. Teknolojia inaweza kutoa sauti za asili katika lugha nyingi tofauti.",
    ),
}


# Longer test texts for more detailed benchmarking (optional)
MULTILINGUAL_LONG_TEXTS = {
    "en": (
        "English",
        """Artificial intelligence has made remarkable progress in natural language processing.
        Modern text-to-speech systems can now generate highly natural and expressive voices in multiple languages.
        This technology enables better accessibility, language learning tools, and immersive content experiences.
        The ability to clone voices across languages opens new possibilities for global communication.""",
    ),
    "es": (
        "Spanish",
        """La inteligencia artificial ha logrado un progreso notable en el procesamiento del lenguaje natural.
        Los sistemas modernos de texto a voz ahora pueden generar voces altamente naturales y expresivas en múltiples idiomas.
        Esta tecnología permite una mejor accesibilidad, herramientas de aprendizaje de idiomas y experiencias de contenido inmersivas.
        La capacidad de clonar voces en diferentes idiomas abre nuevas posibilidades para la comunicación global.""",
    ),
    "fr": (
        "French",
        """L'intelligence artificielle a fait des progrès remarquables dans le traitement du langage naturel.
        Les systèmes modernes de synthèse vocale peuvent désormais générer des voix très naturelles et expressives dans plusieurs langues.
        Cette technologie permet une meilleure accessibilité, des outils d'apprentissage des langues et des expériences de contenu immersives.
        La capacité de cloner des voix dans différentes langues ouvre de nouvelles possibilités pour la communication mondiale.""",
    ),
    "de": (
        "German",
        """Künstliche Intelligenz hat bemerkenswerte Fortschritte in der Verarbeitung natürlicher Sprache gemacht.
        Moderne Text-zu-Sprache-Systeme können jetzt hochnatürliche und ausdrucksstarke Stimmen in mehreren Sprachen erzeugen.
        Diese Technologie ermöglicht bessere Zugänglichkeit, Sprachwerkzeuge und immersive Inhaltserlebnisse.
        Die Fähigkeit, Stimmen über Sprachen hinweg zu klonen, eröffnet neue Möglichkeiten für die globale Kommunikation.""",
    ),
    "ja": (
        "Japanese",
        """人工知能は自然言語処理において顕著な進歩を遂げました。
        最新のテキスト読み上げシステムは、複数の言語で非常に自然で表現力豊かな音声を生成できるようになりました。
        この技術により、より良いアクセシビリティ、言語学習ツール、没入型コンテンツ体験が可能になります。
        言語を超えて音声をクローンする能力は、グローバルコミュニケーションの新しい可能性を開きます。""",
    ),
    "zh": (
        "Chinese",
        """人工智能在自然语言处理方面取得了显著进展。
        现代文本转语音系统现在可以生成多种语言的高度自然和富有表现力的声音。
        这项技术实现了更好的可访问性、语言学习工具和沉浸式内容体验。
        跨语言克隆声音的能力为全球交流开辟了新的可能性。""",
    ),
}


# ============================================================================
# Utility Functions
# ============================================================================

def get_memory_mb() -> float:
    """Get current system memory usage in GB, converted to MB for compatibility."""
    info = get_memory_info()
    return info['sys_used_gb'] * 1024


def get_gpu_memory_mb(device: str) -> Optional[float]:
    """Get GPU memory usage if available."""
    info = get_memory_info()

    if device == "mps" and 'mps_allocated_mb' in info:
        return info['mps_allocated_mb']
    elif device == "cuda" and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return None


def clear_memory(device: str):
    """Clear GPU memory and run garbage collection."""
    # Multiple rounds of GC for thorough cleanup
    for _ in range(3):
        gc.collect()

    if device in ["mlx", "mlx-q4"]:
        # MLX doesn't require explicit cache clearing
        pass
    elif device == "mps" and torch.backends.mps.is_available():
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


def transcribe_audio_whisper(audio_path: str, language: str = "en") -> Optional[str]:
    """
    Transcribe audio using MLX Whisper.
    
    Args:
        audio_path: Path to audio file
        language: Language code for transcription hint
        
    Returns:
        Transcription text or None if failed
    """
    try:
        import mlx_whisper
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo="mlx-community/whisper-base-mlx-q4",
            language=language if language in ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "zh", "ko"] else None,
        )
        return result["text"].strip()
    except ImportError:
        print("⚠️  mlx_whisper not installed. Install with: pip install mlx-whisper")
        return None
    except Exception as e:
        print(f"⚠️  Transcription failed: {e}")
        return None


def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER) between reference and hypothesis.
    
    Args:
        reference: Reference text
        hypothesis: Hypothesis text from transcription
        
    Returns:
        WER as a float (0.0 = perfect match, 1.0 = all errors)
    """
    import numpy as np
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Dynamic programming for Levenshtein distance
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32)
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j] + 1,    # deletion
                             d[i][j-1] + 1,    # insertion
                             d[i-1][j-1] + 1)  # substitution
    
    wer = d[len(ref_words)][len(hyp_words)] / max(len(ref_words), 1)
    return wer


# ============================================================================
# Benchmark Result Classes
# ============================================================================

@dataclass
class MultilingualRunResult:
    """Result from a single generation run."""
    device: str
    language_code: str
    language_name: str
    text_length_words: int
    text_length_chars: int
    generation_time_seconds: float
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    gpu_memory_mb: Optional[float]
    audio_duration_seconds: float
    realtime_factor: float  # audio_duration / generation_time
    wav: Optional[Any] = None


@dataclass
class MultilingualBenchmarkResult:
    """Aggregated result from multiple benchmark runs."""
    device: str
    language_code: str
    language_name: str
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
    
    # Transcription validation (optional)
    transcription: Optional[str] = None
    word_error_rate: Optional[float] = None


# ============================================================================
# Benchmark Runner
# ============================================================================

class MultilingualBenchmark:
    """Benchmark runner for multilingual Chatterbox TTS."""

    def __init__(self, config: MultilingualBenchmarkConfig):
        self.config = config
        self.results: List[MultilingualBenchmarkResult] = []
        self.model = None
        self.current_device = None

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Validate audio prompt is provided
        if not config.audio_prompt_path:
            print("⚠️  Warning: No reference audio provided. Voice cloning will use default voice.")
        elif not Path(config.audio_prompt_path).exists():
            raise FileNotFoundError(f"Reference audio not found: {config.audio_prompt_path}")

    def load_model(self, device: str):
        """Load the multilingual TTS model on specified device.

        Note: Float16 KV cache optimization is enabled by default for PyTorch models.
        MLX models use native optimizations, MLX-Q4 uses 4-bit quantization.
        """
        print(f"\n{'='*60}")
        print(f"Loading multilingual model on {device.upper()}...")

        # Clear any existing model
        if self.model is not None:
            del self.model
            self.model = None
            clear_memory(self.current_device or "cpu")

        is_mlx = device in ["mlx", "mlx-q4"]
        is_hybrid_mlx = device == "hybrid-mlx"

        if is_hybrid_mlx:
            # Hybrid MLX: T3 MLX + S3Gen PyTorch
            print(f"  Backend: Hybrid MLX (T3 MLX + S3Gen PyTorch)")
            print(f"  T3: MLX (Apple Silicon optimized)")
            print(f"  S3Gen: PyTorch on MPS")
        elif is_mlx:
            # MLX models
            print(f"  Backend: MLX (Apple Silicon optimized)")
            if device == "mlx-q4":
                print(f"  Quantization: 4-bit (group_size=64)")
            else:
                print(f"  Precision: Full (float32)")
        else:
            # PyTorch models
            print(f"  Backend: PyTorch")
            print(f"  Float16 KV cache optimization: ENABLED")

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

        print(f"{'='*60}")

        self.current_device = device
        mem_before = get_memory_mb()
        load_start = time.time()

        # Load model based on backend
        if is_hybrid_mlx:
            # Hybrid MLX backend: T3 MLX + S3Gen PyTorch
            try:
                from chatterbox.mtl_tts_mlx import ChatterboxMultilingualTTSMLX
                self.model = ChatterboxMultilingualTTSMLX.from_pretrained()
            except (ImportError, RuntimeError, AttributeError) as e:
                print(f"\n⚠️  Hybrid MLX backend not available")
                print(f"    Error: {str(e)[:100]}")
                raise RuntimeError(f"Hybrid MLX backend not available: {e}")
        elif is_mlx:
            # MLX backend - check if fully implemented
            try:
                from chatterbox.mtl_tts_mlx import ChatterboxMultilingualTTSMLX
                self.model = ChatterboxMultilingualTTSMLX.from_pretrained()

                # Apply quantization if requested
                if device == "mlx-q4":
                    print("  Applying 4-bit quantization...")
                    from chatterbox.models.t3_mlx.quantization.quantize_mlx import QuantizedT3MLX
                    # Quantize the underlying T3 model
                    self.model.model = QuantizedT3MLX(self.model.model, bits=4, group_size=64).model
                    print("  ✓ Quantization complete")
            except (ImportError, RuntimeError, AttributeError) as e:
                print(f"\n⚠️  MLX backend not fully implemented yet")
                print(f"    Error: {str(e)[:100]}")
                print(f"    Skipping MLX benchmarks...")
                print(f"    Note: Currently only PyTorch backends (mps, cpu) are supported")
                raise RuntimeError(f"MLX backend not available: {e}")
        else:
            # PyTorch backend
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            self.model = ChatterboxMultilingualTTS.from_pretrained(device=device)

        load_time = time.time() - load_start
        mem_after = get_memory_mb()

        print(f"✓ Model loaded in {load_time:.2f}s")
        print(f"  Memory: {mem_before:.1f} → {mem_after:.1f} MB (+{mem_after - mem_before:.1f} MB)")

        # Prepare conditionals if audio prompt provided
        if self.config.audio_prompt_path:
            print(f"  Loading voice from: {self.config.audio_prompt_path}")
            self.model.prepare_conditionals(self.config.audio_prompt_path)
            print("  ✓ Voice conditionals prepared")

    def run_single_generation(
        self,
        text: str,
        language_id: str,
        language_name: str,
        run_id: int = 0,
        keep_wav: bool = False
    ) -> MultilingualRunResult:
        """Run a single TTS generation and collect metrics."""
        clear_memory(self.current_device)

        mem_before = get_memory_mb()
        gpu_mem_before = get_gpu_memory_mb(self.current_device)

        start_time = time.time()

        # Generate audio
        wav = self.model.generate(
            text,
            language_id=language_id,
            audio_prompt_path=self.config.audio_prompt_path if run_id == 0 else None,
            exaggeration=0.5,
            cfg_weight=0.5,
            show_progress=False,  # Suppress tqdm for clean benchmark output
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

        result = MultilingualRunResult(
            device=self.current_device,
            language_code=language_id,
            language_name=language_name,
            text_length_words=word_count(text),
            text_length_chars=char_count(text),
            generation_time_seconds=generation_time,
            memory_before_mb=mem_before,
            memory_after_mb=mem_after,
            memory_delta_mb=mem_after - mem_before,
            gpu_memory_mb=gpu_mem_after,
            audio_duration_seconds=audio_duration,
            realtime_factor=realtime_factor,
            wav=wav if keep_wav else None,
        )

        # If caller does not want to keep the waveform, delete it and clear memory.
        if not keep_wav:
            del wav
            clear_memory(self.current_device)

        return result

    def benchmark_language(self, language_code: str, text: str, language_name: str) -> MultilingualBenchmarkResult:
        """Run complete benchmark on a single language."""
        text_words = word_count(text)
        text_chars = char_count(text)
        preview = text[:50] + "..." if len(text) > 50 else text

        print(f"\n  Testing: {language_name} ({language_code})")
        print(f"  Text: {preview}")
        print(f"  Length: {text_words} words, {text_chars} chars")

        # Warmup runs
        print(f"  Warmup runs: ", end="", flush=True)
        for i in range(self.config.warmup_runs):
            self.run_single_generation(text, language_code, language_name, run_id=i)
            print(".", end="", flush=True)
        print(" done")

        # Benchmark runs
        print(f"  Benchmark runs: ", end="", flush=True)
        run_results: List[MultilingualRunResult] = []
        for i in range(self.config.benchmark_runs):
            # Keep the waveform from the last timed run if we want to save audio
            keep = self.config.save_audio and (i == self.config.benchmark_runs - 1)
            result = self.run_single_generation(
                text, language_code, language_name,
                run_id=i + self.config.warmup_runs,
                keep_wav=keep
            )
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

        result = MultilingualBenchmarkResult(
            device=self.current_device,
            language_code=language_code,
            language_name=language_name,
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
        print(f"  → Throughput: {result.words_per_second:.1f} words/s")

        # Save the last waveform immediately if requested
        output_path = None
        if self.config.save_audio:
            last_wav = run_results[-1].wav
            if last_wav is not None:
                output_path = Path(self.config.output_dir) / f"{self.current_device}_{language_code}.wav"

                # Handle both torch tensors (PyTorch) and numpy arrays (MLX)
                if isinstance(last_wav, torch.Tensor):
                    # Already a torch tensor
                    wav_tensor = last_wav.float()
                else:
                    # Numpy array (from MLX), convert to torch for torchaudio.save
                    wav_tensor = torch.from_numpy(last_wav).float()
                
                ta.save(str(output_path), wav_tensor, self.model.sr)

                print(f"  → Saved: {output_path}")
                # Remove reference from the stored result so memory can be freed
                run_results[-1].wav = None
                del last_wav
                clear_memory(self.current_device)
        
        # Whisper transcription validation
        if self.config.validate_transcription and output_path and output_path.exists():
            print(f"  Validating transcription...")
            transcription = transcribe_audio_whisper(str(output_path), language=language_code)
            if transcription:
                wer = word_error_rate(text, transcription)
                result.transcription = transcription
                result.word_error_rate = wer
                preview_trans = transcription[:60] + "..." if len(transcription) > 60 else transcription
                print(f"  → Transcription: '{preview_trans}'")
                print(f"  → Word Error Rate: {wer:.2%}")

        # Force cleanup between benchmarks
        clear_memory(self.current_device)

        return result

    def run_benchmarks(self):
        """Run all benchmarks."""
        print("\n" + "="*70)
        print("CHATTERBOX MULTILINGUAL TTS BENCHMARK")
        print("="*70)
        print(f"Platform: macOS with Apple Silicon")
        print(f"MPS Available: {torch.backends.mps.is_available()}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Warmup Runs: {self.config.warmup_runs}")
        print(f"Benchmark Runs: {self.config.benchmark_runs}")
        print(f"Devices to Test: {', '.join(self.config.test_devices)}")
        print(f"Languages to Test: {', '.join(self.config.test_languages)}")
        if self.config.audio_prompt_path:
            print(f"Reference Audio: {self.config.audio_prompt_path}")
        if self.config.validate_transcription:
            print(f"Transcription Validation: ENABLED (MLX Whisper)")
        print(f"\n✓ Float16 KV Cache: ENABLED (default)")
        print(f"  - Expected: 18-32% faster generation")
        print(f"  - Expected: Significant memory savings")
        print("="*70)

        for device in self.config.test_devices:
            try:
                self.load_model(device)
            except RuntimeError as e:
                # Skip devices that aren't available (e.g., MLX not fully implemented)
                if "not available" in str(e) or "not fully implemented" in str(e):
                    print(f"\n⚠️  Skipping {device.upper()} benchmarks (not available)")
                    continue
                else:
                    raise

            print(f"\n{'─'*60}")
            print(f"BENCHMARKING ON {device.upper()}")
            print(f"{'─'*60}")

            for lang_code in self.config.test_languages:
                if lang_code not in MULTILINGUAL_TEXTS:
                    print(f"⚠️  Skipping {lang_code}: No test text available")
                    continue

                lang_name, text = MULTILINGUAL_TEXTS[lang_code]
                result = self.benchmark_language(lang_code, text, lang_name)
                self.results.append(result)

    def print_summary(self):
        """Print benchmark summary table."""
        print("\n" + "="*100)
        print("MULTILINGUAL BENCHMARK SUMMARY")
        print("="*100)
        print("\nNote: PyTorch results use float16 KV cache optimization (enabled by default)")
        print("      MLX results use native optimizations, MLX-Q4 uses 4-bit quantization")
        print("="*100)

        # Group results by device
        devices = list(set(r.device for r in self.results))
        languages = sorted(set(r.language_code for r in self.results))

        # Header
        print(f"\n{'Language':<20} ", end="")
        for device in devices:
            print(f"│ {device.upper():<35} ", end="")
        print()

        print(f"{'─'*20} ", end="")
        for device in devices:
            print(f"│ {'─'*35} ", end="")
        print()

        # Data rows
        for lang_code in languages:
            lang_results = [r for r in self.results if r.language_code == lang_code]
            if not lang_results:
                continue

            lang_name = lang_results[0].language_name
            lang_display = f"{lang_name} ({lang_code})"
            print(f"{lang_display:<20} ", end="")

            for device in devices:
                dev_result = next((r for r in lang_results if r.device == device), None)
                if dev_result:
                    time_str = f"{dev_result.mean_time:.2f}s"
                    rtf_str = f"{dev_result.mean_realtime_factor:.2f}x RTF"
                    print(f"│ {time_str:<10} {rtf_str:<23} ", end="")
                else:
                    print(f"│ {'N/A':<35} ", end="")
            print()

        # Speedup comparisons
        if len(devices) > 1:
            print(f"\n{'─'*100}")
            print("Performance Comparisons by Language:")

            # Define baseline as MPS or CPU if MPS not available
            baseline_device = "mps" if "mps" in devices else ("cpu" if "cpu" in devices else devices[0])

            for lang_code in languages:
                lang_results = [r for r in self.results if r.language_code == lang_code]
                if not lang_results:
                    continue

                baseline_result = next((r for r in lang_results if r.device == baseline_device), None)
                if not baseline_result or baseline_result.mean_time == 0:
                    continue

                lang_name = baseline_result.language_name
                print(f"\n  {lang_name} ({lang_code}) - vs {baseline_device.upper()}:")

                for device in devices:
                    if device == baseline_device:
                        continue

                    dev_result = next((r for r in lang_results if r.device == device), None)
                    if dev_result and dev_result.mean_time > 0:
                        speedup = baseline_result.mean_time / dev_result.mean_time
                        if speedup > 1:
                            print(f"    {device.upper():<10}: {speedup:.2f}x faster")
                        else:
                            print(f"    {device.upper():<10}: {1/speedup:.2f}x slower")

        # Average performance
        print(f"\n{'─'*100}")
        print("Average Performance Across Languages:")
        for device in devices:
            dev_results = [r for r in self.results if r.device == device]
            if dev_results:
                avg_time = statistics.mean(r.mean_time for r in dev_results)
                avg_rtf = statistics.mean(r.mean_realtime_factor for r in dev_results)
                print(f"  {device.upper():<10}: {avg_time:.2f}s avg time, {avg_rtf:.2f}x avg RTF")

        # Transcription validation summary
        results_with_wer = [r for r in self.results if r.word_error_rate is not None]
        if results_with_wer:
            print(f"\n{'─'*100}")
            print("Transcription Validation (Word Error Rate):")
            # Group by device
            for device in devices:
                device_wer_results = [r for r in results_with_wer if r.device == device]
                if device_wer_results:
                    avg_wer = statistics.mean(r.word_error_rate for r in device_wer_results)
                    print(f"\n  {device.upper()}:")
                    for r in device_wer_results:
                        print(f"    {r.language_name:<15} ({r.language_code}): {r.word_error_rate:.2%} WER")
                    print(f"    {'Average':<15}      : {avg_wer:.2%} WER")
        
        print(f"\n{'='*100}")

    def export_results(self, filepath: str = None):
        """Export results to JSON."""
        import json

        if filepath is None:
            filepath = Path(self.config.output_dir) / "multilingual_results.json"

        data = {
            "config": {
                "warmup_runs": self.config.warmup_runs,
                "benchmark_runs": self.config.benchmark_runs,
                "audio_prompt_path": self.config.audio_prompt_path,
                "test_devices": self.config.test_devices,
                "test_languages": self.config.test_languages,
            },
            "system": {
                "mps_available": torch.backends.mps.is_available(),
                "pytorch_version": torch.__version__,
                "total_ram_gb": psutil.virtual_memory().total / (1024**3),
            },
            "results": [
                {
                    "device": r.device,
                    "language_code": r.language_code,
                    "language_name": r.language_name,
                    "text_preview": r.text_preview,
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

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\nResults exported to: {filepath}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the multilingual benchmark."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Chatterbox Multilingual TTS Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported Languages:
  en (English), es (Spanish), fr (French), de (German), it (Italian),
  pt (Portuguese), ru (Russian), ja (Japanese), zh (Chinese), ko (Korean),
  ar (Arabic), hi (Hindi), tr (Turkish), pl (Polish), nl (Dutch),
  sv (Swedish), da (Danish), no (Norwegian), fi (Finnish), el (Greek),
  he (Hebrew), ms (Malay), sw (Swahili)

Example:
  python benchmark_multilingual.py --audio-prompt speaker.wav --languages en es fr ja zh
        """
    )
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--devices", nargs="+", default=["mps", "cpu", "hybrid-mlx", "mlx", "mlx-q4"],
                       help="Devices to benchmark (mps, cpu, hybrid-mlx, mlx, mlx-q4). MLX will be skipped if not available.")
    parser.add_argument("--audio-prompt", type=str, required=True,
                       help="Path to reference audio for voice cloning (REQUIRED)")
    parser.add_argument("--languages", nargs="+", default=["en", "es", "fr", "de", "ja", "zh"],
                       help="Language codes to test (default: en es fr de ja zh)")
    parser.add_argument("--output-dir", type=str, default="benchmark_multilingual_output",
                       help="Output directory for results")
    parser.add_argument("--no-save-audio", action="store_true",
                       help="Don't save generated audio files")
    parser.add_argument("--validate", action="store_true",
                       help="Validate generated audio using MLX Whisper transcription and compute WER")
    parser.add_argument("--mps-only", action="store_true",
                       help="Only benchmark MPS (skip others)")
    parser.add_argument("--cpu-only", action="store_true",
                       help="Only benchmark CPU (skip others)")
    parser.add_argument("--hybrid-mlx-only", action="store_true",
                       help="Only benchmark Hybrid MLX (T3 MLX + S3Gen PyTorch)")
    parser.add_argument("--mlx-only", action="store_true",
                       help="Only benchmark MLX full precision (skip others)")
    parser.add_argument("--mlx-q4-only", action="store_true",
                       help="Only benchmark MLX quantized (skip others)")

    args = parser.parse_args()

    # Validate languages
    invalid_langs = [lang for lang in args.languages if lang not in MULTILINGUAL_TEXTS]
    if invalid_langs:
        print(f"Error: Invalid language codes: {', '.join(invalid_langs)}")
        print(f"Supported languages: {', '.join(sorted(MULTILINGUAL_TEXTS.keys()))}")
        return 1

    # Handle device selection
    if args.mps_only:
        devices = ["mps"]
    elif args.cpu_only:
        devices = ["cpu"]
    elif args.hybrid_mlx_only:
        devices = ["hybrid-mlx"]
    elif args.mlx_only:
        devices = ["mlx"]
    elif args.mlx_q4_only:
        devices = ["mlx-q4"]
    else:
        devices = args.devices

    config = MultilingualBenchmarkConfig(
        warmup_runs=args.warmup,
        benchmark_runs=args.runs,
        test_devices=devices,
        audio_prompt_path=args.audio_prompt,
        test_languages=args.languages,
        output_dir=args.output_dir,
        save_audio=not args.no_save_audio,
        validate_transcription=args.validate,
    )

    benchmark = MultilingualBenchmark(config)

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
    exit(main() or 0)
