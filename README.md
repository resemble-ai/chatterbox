# Chatterbox MLX - Apple Silicon Optimized TTS

[![PyPI version](https://badge.fury.io/py/chatterbox-mlx.svg)](https://badge.fury.io/py/chatterbox-mlx)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An MLX-optimized fork of [Resemble AI's Chatterbox TTS](https://github.com/resemble-ai/chatterbox) for Apple Silicon, delivering up to 2.4x faster inference.**

---

## Installation

```bash
pip install chatterbox-mlx
```

### Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- ~4GB disk space for model weights

---

## CLI Usage

Generate speech directly from the terminal:

```bash
    # Generate English speech (auto-generated filename):
    chatterbox "Artificial intelligence has made remarkable strides in recent years, particularly in the field of natural language processing."

    # Generate Spanish speech:
    chatterbox "La inteligencia artificial ha logrado avances notables en los últimos años." --lang es

    # Use the --voice flag to provide a reference audio file for voice cloning:
    chatterbox "Artificial intelligence has made remarkable strides in recent years, particularly in the field of natural language processing." --voice speaker.wav

    # Run multilingual benchmark (saves to benchmark_output/)
    chatterbox --benchmark --languages en es
```

### CLI Options

| Option            | Description                                  | Default           |
| ----------------- | -------------------------------------------- | ----------------- |
| `-o, --output`    | Output WAV file path                         | Auto-generated    |
| `-l, --lang`      | Language code (en, es, fr, de, ja, zh, etc.) | `en`              |
| `-v, --voice`     | Reference audio for voice cloning            | None              |
| `--exaggeration`  | Emotion intensity (0.0-1.0)                  | `0.5`             |
| `--cfg`           | Classifier-free guidance weight              | `0.5`             |
| `--backend`       | Backend: hybrid-mlx, mlx, pytorch            | `hybrid-mlx`      |
| `--benchmark`     | Run multilingual benchmark                   | False             |
| `--languages`     | Languages to benchmark                       | en es fr de ja zh |
| `--no-save-audio` | Don't save benchmark audio files             | False (saves)     |
| `-q, --quiet`     | Suppress progress messages                   | False             |

---

## Quick Start

```python
import torchaudio as ta
from chatterbox.tts_mlx import ChatterboxTTSMLX

# Load model (downloads weights automatically on first run)
model = ChatterboxTTSMLX.from_pretrained(device="mps")

# Generate speech
text = "Hello! This is Chatterbox running with MLX optimization on Apple Silicon."
wav = model.generate(text)
ta.save("output.wav", wav, model.sr)

# Voice cloning with reference audio
wav = model.generate(
    text,
    audio_prompt_path="reference_voice.wav",
    exaggeration=0.5,  # Emotion intensity (0.0-1.0)
    cfg_weight=0.5,    # Classifier-free guidance
)
```

### Long-Form Audio Generation

For texts longer than ~50 words, use chunked generation:

```python
long_text = """
Your long text here. It can span multiple paragraphs and sentences.
The generate_long method will automatically split it at sentence boundaries,
generate each chunk separately, and crossfade them together seamlessly.
"""

wav = model.generate_long(
    long_text,
    audio_prompt_path="reference_voice.wav",
    chunk_size_words=50,
    overlap_duration=0.1,
)
ta.save("long_output.wav", wav, model.sr)
```

## 🙏 Acknowledgements

This project is built on top of the excellent **[Chatterbox TTS](https://github.com/resemble-ai/chatterbox)** by [Resemble AI](https://resemble.ai). I'm deeply grateful for their work in creating and open-sourcing a production-grade, multilingual text-to-speech system under the MIT license.

**This fork focuses specifically on MLX optimizations for Apple Silicon.** If you're looking for the original project with CUDA support and the full feature set, please visit the [official Resemble AI repository](https://github.com/resemble-ai/chatterbox).

---

## What's Different in This Fork?

This package provides **native MLX acceleration** for Apple Silicon Macs, achieving significant performance improvements:

| Text Length       | CPU Baseline | MLX Optimized | Speedup         |
| ----------------- | ------------ | ------------- | --------------- |
| Short (5 words)   | 8.91s        | 3.70s         | **2.4x faster** |
| Medium (31 words) | 57.51s       | 24.40s        | **2.4x faster** |
| Long (94 words)   | 137.92s      | 62.66s        | **2.2x faster** |

### Key Optimizations

- **MLX-Native T3 Model**: The 520M parameter Llama 3 backbone runs entirely on MLX
- **Float16 KV Cache**: Up to 5.8 GB memory savings with 32% faster generation
- **Hybrid Architecture**: Combines MLX speed with PyTorch quality controls
- **Long-Form Generation**: Intelligent chunking with crossfade for extended audio

---

## Benchmark Results

All benchmarks run on **Apple M4 (32GB RAM), macOS 15.4, Python 3.11, PyTorch 2.8.0**.

### English TTS Performance

| Device         | Text   | Words | Time    | RTF   |
| -------------- | ------ | ----- | ------- | ----- |
| **Hybrid-MLX** | short  | 5     | 4.08s   | 0.65x |
| **Hybrid-MLX** | medium | 31    | 25.24s  | 0.73x |
| **Hybrid-MLX** | long   | 94    | 62.66s  | 0.74x |
| Pure MLX       | short  | 5     | 3.70s   | 0.69x |
| Pure MLX       | medium | 31    | 24.40s  | 0.72x |
| Pure MLX       | long   | 94    | 68.82s  | 0.71x |
| CPU            | short  | 5     | 8.91s   | 0.27x |
| CPU            | medium | 31    | 57.51s  | 0.33x |
| CPU            | long   | 94    | 137.92s | 0.34x |

**Key findings:**

- **Hybrid-MLX** recommended for production (best quality/speed balance)
- **Pure MLX** fastest for short texts, but quality degrades on long texts
- **2.2-2.4x speedup** vs CPU baseline across all text lengths

### Multilingual Performance

| Device     | Language | Time   | RTF   |
| ---------- | -------- | ------ | ----- |
| Hybrid-MLX | English  | 12.25s | 0.71x |
| Hybrid-MLX | Spanish  | 14.74s | 0.76x |
| Pure MLX   | English  | 14.55s | 0.67x |
| Pure MLX   | Spanish  | 13.78s | 0.75x |
| MPS        | English  | 19.96s | 0.50x |
| MPS        | Spanish  | 21.06s | 0.51x |
| CPU        | English  | 25.64s | 0.32x |
| CPU        | Spanish  | 31.31s | 0.33x |

### Visual Comparison

```
                    GENERATION TIME COMPARISON

     Short (5 words)
     ├─ CPU        ████████████████████████████████████████ 8.91s
     ├─ Hybrid-MLX ██████████████████ 4.08s (2.2x faster)
     └─ Pure MLX   ████████████████ 3.70s (2.4x faster)

     Medium (31 words)
     ├─ CPU        ████████████████████████████████████████ 57.51s
     ├─ Hybrid-MLX █████████████████ 25.24s (2.3x faster)
     └─ Pure MLX   ████████████████ 24.40s (2.4x faster)

     Long (94 words)
     ├─ CPU        ████████████████████████████████████████ 137.92s
     ├─ Hybrid-MLX █████████████████ 62.66s (2.2x faster)  ✓ Best quality
     └─ Pure MLX   ██████████████████ 68.82s (2.0x faster)
```

### Backend Comparison

| Backend         | Description                    | RTF   | Memory | Recommendation       |
| --------------- | ------------------------------ | ----- | ------ | -------------------- |
| **Hybrid-MLX**  | T3 (MLX) + S3Gen (PyTorch/MPS) | 0.74x | ~16GB  | ✅ Production use    |
| **Pure MLX**    | Everything on MLX              | 0.71x | ~14GB  | Minimal dependencies |
| **PyTorch MPS** | Full PyTorch on MPS            | 0.51x | ~14GB  | Fallback             |
| **CPU**         | PyTorch on CPU                 | 0.34x | ~14GB  | Baseline             |

_RTF = Real-Time Factor (audio_duration / generation_time). Higher is better._

---

## Running Benchmarks

You can reproduce these benchmarks on your own hardware.

### English TTS Benchmark

```bash
# Full benchmark (all backends)
python benchmark_mps.py --runs 3 --validate

# Quick test with Hybrid-MLX only
python benchmark_mps.py --hybrid-mlx-only --runs 1

# CPU baseline only
python benchmark_mps.py --cpu-only --runs 1

# With voice cloning
python benchmark_mps.py --audio-prompt speaker.wav --runs 3

# Enable memory debugging
DEBUG_MEMORY=1 python benchmark_mps.py --hybrid-mlx-only
```

**Options:**
| Flag | Description |
|------|-------------|
| `--warmup N` | Warmup runs before timing (default: 1) |
| `--runs N` | Number of timed benchmark runs (default: 3) |
| `--devices` | Backends to test: `mps`, `cpu`, `hybrid-mlx`, `mlx`, `mlx-q4` |
| `--audio-prompt FILE` | Reference audio for voice cloning |
| `--output-dir DIR` | Output directory (default: `benchmark_output/`) |
| `--validate` | Enable Whisper transcription validation (computes WER) |
| `--mps-only` | Only benchmark PyTorch MPS |
| `--cpu-only` | Only benchmark CPU |
| `--hybrid-mlx-only` | Only benchmark Hybrid-MLX |
| `--mlx-only` | Only benchmark Pure MLX |
| `--debug-memory` | Enable detailed memory logging |

### Multilingual Benchmark

```bash
# Test specific languages
python benchmark_multilingual.py \
    --audio-prompt speaker.wav \
    --languages en es fr de ja zh \
    --runs 3

# Quick test with Hybrid-MLX
python benchmark_multilingual.py \
    --audio-prompt speaker.wav \
    --languages en es \
    --hybrid-mlx-only

# With validation
python benchmark_multilingual.py \
    --audio-prompt speaker.wav \
    --languages en es fr \
    --validate
```

**Supported Languages:**
`en` (English), `es` (Spanish), `fr` (French), `de` (German), `it` (Italian), `pt` (Portuguese), `ru` (Russian), `ja` (Japanese), `zh` (Chinese), `ko` (Korean), `ar` (Arabic), `hi` (Hindi), `tr` (Turkish), `pl` (Polish), `nl` (Dutch), `sv` (Swedish), `da` (Danish), `no` (Norwegian), `fi` (Finnish), `el` (Greek), `he` (Hebrew), `ms` (Malay), `sw` (Swahili)

### Benchmark Output

Results are saved to:

- `benchmark_output/benchmark_results.json` - English TTS results
- `benchmark_multilingual_output/multilingual_results.json` - Multilingual results
- Generated audio files: `{device}_{category}.wav`

---

## Architecture

Chatterbox is a two-stage TTS pipeline. This fork accelerates the most compute-intensive component (T3) with MLX:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CHATTERBOX MLX PIPELINE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │ VoiceEncoder │    │     T3       │    │       S3Gen          │   │
│  │  (PyTorch)   │───▶│    (MLX)     │───▶│   (PyTorch/MPS)      │   │
│  │   ~2M params │    │  520M params │    │     ~80M params      │   │
│  └──────────────┘    └──────────────┘    └──────────────────────┘   │
│                            ▲                                        │
│                            │                                        │
│                    2.4x faster with MLX                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Supported Languages

All 23 languages from the original Chatterbox are supported:

Arabic • Danish • German • Greek • English • Spanish • Finnish • French • Hebrew • Hindi • Italian • Japanese • Korean • Malay • Dutch • Norwegian • Polish • Portuguese • Russian • Swedish • Swahili • Turkish • Chinese

```python
from chatterbox.mtl_tts_mlx import ChatterboxMultilingualTTSMLX

model = ChatterboxMultilingualTTSMLX.from_pretrained(device="mps")

# French
wav = model.generate("Bonjour, comment ça va?", language_id="fr")

# Japanese
wav = model.generate("こんにちは、元気ですか？", language_id="ja")
```

---

## Tips for Best Results

### General Use

- Default settings (`exaggeration=0.5`, `cfg_weight=0.5`) work well for most cases
- Ensure reference audio matches target language to avoid accent transfer

### Expressive Speech

- Lower `cfg_weight` (~0.3) + higher `exaggeration` (~0.7) for dramatic delivery
- Higher exaggeration speeds up speech; lower cfg_weight compensates

### Memory Usage

Enable debug logging to monitor memory:

```bash
DEBUG_MEMORY=1 python your_script.py
```

---

## Differences from Original Chatterbox

| Feature             | Original (Resemble AI) | This Fork              |
| ------------------- | ---------------------- | ---------------------- |
| **Target Hardware** | NVIDIA CUDA            | Apple Silicon          |
| **ML Framework**    | PyTorch                | MLX + PyTorch hybrid   |
| **T3 Inference**    | PyTorch                | MLX (2.4x faster)      |
| **KV Cache**        | Float32                | Float16 (32% faster)   |
| **Long-form Audio** | Basic                  | Chunked with crossfade |

---

## Credits & Links

- **Original Project**: [Resemble AI's Chatterbox](https://github.com/resemble-ai/chatterbox)
- **Resemble AI**: [resemble.ai](https://resemble.ai) - For creating and open-sourcing this incredible TTS system
- **Demo**: [Hugging Face Space](https://huggingface.co/spaces/ResembleAI/Chatterbox)
- **Evaluation**: [Outperforms ElevenLabs](https://podonos.com/resembleai/chatterbox)

### Upstream Dependencies

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [HiFT-GAN](https://github.com/yl4579/HiFTNet)
- [Llama 3](https://github.com/meta-llama/llama3)
- [MLX](https://github.com/ml-explore/mlx)

---

## License

MIT License - Same as the original Chatterbox project.

---

## Citation

If you use this project, please cite the original Chatterbox:

```bibtex
@misc{chatterboxtts2025,
  author       = {{Resemble AI}},
  title        = {{Chatterbox-TTS}},
  year         = {2025},
  howpublished = {\url{https://github.com/resemble-ai/chatterbox}},
  note         = {GitHub repository}
}
```
