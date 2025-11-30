<img width="1200" height="600" alt="Chatterbox-Multilingual" src="https://www.resemble.ai/wp-content/uploads/2025/09/Chatterbox-Multilingual-1.png" />

# Chatterbox TTS

[![Alt Text](https://img.shields.io/badge/listen-demo_samples-blue)](https://resemble-ai.github.io/chatterbox_demopage/)
[![Alt Text](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/ResembleAI/Chatterbox)
[![Alt Text](https://static-public.podonos.com/badges/insight-on-pdns-sm-dark.svg)](https://podonos.com/resembleai/chatterbox)
[![Discord](https://img.shields.io/discord/1377773249798344776?label=join%20discord&logo=discord&style=flat)](https://discord.gg/rJq9cRJBJ6)

\_Made with ♥️ by <a href="https://resemble.ai" target="_blank"><img width="100" alt="resemble-logo-horizontal" src="https://github.com/user-attachments/assets/35cf756b-3506-4943-9c72-c05ddfa4e525" /></a>

We're excited to introduce **Chatterbox Multilingual**, [Resemble AI's](https://resemble.ai) first production-grade open source TTS model supporting **23 languages** out of the box. Licensed under MIT, Chatterbox has been benchmarked against leading closed-source systems like ElevenLabs, and is consistently preferred in side-by-side evaluations.

Whether you're working on memes, videos, games, or AI agents, Chatterbox brings your content to life across languages. It's also the first open source TTS model to support **emotion exaggeration control** with robust **multilingual zero-shot voice cloning**. Try the english only version now on our [English Hugging Face Gradio app.](https://huggingface.co/spaces/ResembleAI/Chatterbox). Or try the multilingual version on our [Multilingual Hugging Face Gradio app.](https://huggingface.co/spaces/ResembleAI/Chatterbox-Multilingual-TTS).

If you like the model but need to scale or tune it for higher accuracy, check out our competitively priced TTS service (<a href="https://resemble.ai">link</a>). It delivers reliable performance with ultra-low latency of sub 200ms—ideal for production use in agents, applications, or interactive media.

# Key Details

- Multilingual, zero-shot TTS supporting 23 languages
- SoTA zeroshot English TTS
- 0.5B Llama backbone
- Unique exaggeration/intensity control
- Ultra-stable with alignment-informed inference
- Trained on 0.5M hours of cleaned data
- Watermarked outputs
- Easy voice conversion script
- [Outperforms ElevenLabs](https://podonos.com/resembleai/chatterbox)

# Supported Languages

Arabic (ar) • Danish (da) • German (de) • Greek (el) • English (en) • Spanish (es) • Finnish (fi) • French (fr) • Hebrew (he) • Hindi (hi) • Italian (it) • Japanese (ja) • Korean (ko) • Malay (ms) • Dutch (nl) • Norwegian (no) • Polish (pl) • Portuguese (pt) • Russian (ru) • Swedish (sv) • Swahili (sw) • Turkish (tr) • Chinese (zh)

# Tips

- **General Use (TTS and Voice Agents):**

  - Ensure that the reference clip matches the specified language tag. Otherwise, language transfer outputs may inherit the accent of the reference clip’s language. To mitigate this, set `cfg_weight` to `0`.
  - The default settings (`exaggeration=0.5`, `cfg_weight=0.5`) work well for most prompts across all languages.
  - If the reference speaker has a fast speaking style, lowering `cfg_weight` to around `0.3` can improve pacing.

- **Expressive or Dramatic Speech:**
  - Try lower `cfg_weight` values (e.g. `~0.3`) and increase `exaggeration` to around `0.7` or higher.
  - Higher `exaggeration` tends to speed up speech; reducing `cfg_weight` helps compensate with slower, more deliberate pacing.

# Installation

```shell
pip install chatterbox-tts
```

Alternatively, you can install from source:

```shell
# conda create -yn chatterbox python=3.11
# conda activate chatterbox

git clone https://github.com/resemble-ai/chatterbox.git
cd chatterbox
pip install -e .
```

We developed and tested Chatterbox on Python 3.11 on Debian 11 OS; the versions of the dependencies are pinned in `pyproject.toml` to ensure consistency. You can modify the code or dependencies in this installation mode.

# Usage

```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# English example
model = ChatterboxTTS.from_pretrained(device="cuda")

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
wav = model.generate(text)
ta.save("test-english.wav", wav, model.sr)

# Multilingual examples
multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)

french_text = "Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox, il prend en charge 23 langues."
wav_french = multilingual_model.generate(spanish_text, language_id="fr")
ta.save("test-french.wav", wav_french, model.sr)

chinese_text = "你好，今天天气真不错，希望你有一个愉快的周末。"
wav_chinese = multilingual_model.generate(chinese_text, language_id="zh")
ta.save("test-chinese.wav", wav_chinese, model.sr)

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save("test-2.wav", wav, model.sr)
```

See `example_tts.py` and `example_vc.py` for more examples.

# Performance Optimizations

## MLX Support (Apple Silicon)

Chatterbox supports **MLX** for native Apple Silicon optimization, providing performance improvements on M-series chips.

### MLX Backends

| Backend         | Description                    | RTF\*     | Best For                 |
| --------------- | ------------------------------ | --------- | ------------------------ |
| **MLX Hybrid**  | T3 (MLX) + S3Gen (PyTorch/MPS) | **0.81x** | Production use (fastest) |
| **Pure MLX**    | T3 (MLX) + S3Gen (MLX)         | 0.67x     | Minimal dependencies     |
| **PyTorch MPS** | Full PyTorch on MPS            | 0.57x     | Compatibility            |

\*RTF = Real-Time Factor (audio_duration / processing_time). Higher is better.

### Installation with MLX

```bash
pip install chatterbox-tts[mlx]
```

### Usage

```python
import torchaudio as ta
from chatterbox.tts_mlx import ChatterboxTTSMLX

# Load MLX Hybrid model (recommended - fastest)
model = ChatterboxTTSMLX.from_pretrained(device="mps")

# Basic generation
text = "Hello! This is running with MLX optimization on Apple Silicon."
wav = model.generate(text)
ta.save("output.wav", wav, model.sr)

# Voice cloning
wav = model.generate(
    text,
    audio_prompt_path="reference_voice.wav",
    exaggeration=0.5,  # Emotion intensity (0.0-1.0)
    cfg_weight=0.5,    # Classifier-free guidance
)

# Long-form generation (automatically chunks and crossfades)
long_text = "Your long text here. It can span multiple paragraphs..."
wav = model.generate_long(
    long_text,
    audio_prompt_path="reference_voice.wav",
    chunk_size_words=50,    # Words per chunk
    overlap_duration=0.1,   # Crossfade duration in seconds
)
ta.save("long_output.wav", wav, model.sr)
```

### Pure MLX (No PyTorch Dependency)

If you want to avoid PyTorch entirely:

```python
from chatterbox.tts_mlx import ChatterboxTTSPureMLX

# Load Pure MLX model (slightly slower, but no PyTorch needed)
model = ChatterboxTTSPureMLX.from_pretrained()
wav = model.generate(text)
```

### Quantized Models (4-bit/8-bit)

For even faster inference with minimal quality loss:

```python
from chatterbox.models.t3_mlx.quantization import QuantizedT3MLX

# Load with 4-bit quantization
model = QuantizedT3MLX.from_pretrained(
    ckpt_path="path/to/checkpoint",
    bits=4,  # or 8 for 8-bit quantization
    group_size=64
)
```

### Performance Notes

- **MLX Hybrid is fastest** because PyTorch's MPS backend has highly optimized kernels for vocoder operations (ISTFT, overlap-add)
- **T3 stage** (68% of compute) runs on MLX in both Hybrid and Pure modes
- **S3Gen stage** (32% of compute) benefits from PyTorch's optimized Metal shaders in Hybrid mode

## PyTorch KV Cache Optimization

Chatterbox includes automatic **float16 KV cache optimization** (enabled by default) for PyTorch, providing significant performance improvements:

- **18-32% faster generation** depending on text length
- **Up to 5.8 GB memory savings** for long-form content
- **No quality degradation** - extensively tested

**Benchmark Results (Apple Silicon M4):**
| Text Length | Speed Improvement | Memory Savings |
|-------------|------------------|----------------|
| Long | 31.9% faster | ~5.8 GB |
| Medium | 18.2% faster | ~645 MB |
| Short | 5.1% faster | ~582 MB |

**Advanced Configuration:**

```python
from chatterbox.models.t3.modules.t3_config import T3Config

# Float16 is enabled by default, but you can disable it if needed:
config = T3Config.english_only(kv_cache_dtype=None)  # Disable optimization
model = ChatterboxTTS.from_pretrained(device="cuda", t3_config=config)
```

## Memory Debugging

For debugging memory usage (especially useful for MLX backends), you can enable detailed memory logging:

```bash
# Via environment variable
DEBUG_MEMORY=1 python your_script.py

# Or when running benchmarks
DEBUG_MEMORY=1 python benchmark_mps.py --hybrid-mlx-only
```

This logs memory usage at key points during model loading and inference, including:

- System memory (used, wired, active)
- MPS allocated memory (for PyTorch Metal backend)
- MLX memory synchronization points

# Acknowledgements

- [Cosyvoice](https://github.com/FunAudioLLM/CosyVoice)
- [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
- [HiFT-GAN](https://github.com/yl4579/HiFTNet)
- [Llama 3](https://github.com/meta-llama/llama3)
- [S3Tokenizer](https://github.com/xingchensong/S3Tokenizer)

# Built-in PerTh Watermarking for Responsible AI

Every audio file generated by Chatterbox includes [Resemble AI's Perth (Perceptual Threshold) Watermarker](https://github.com/resemble-ai/perth) - imperceptible neural watermarks that survive MP3 compression, audio editing, and common manipulations while maintaining nearly 100% detection accuracy.

## Watermark extraction

You can look for the watermark using the following script.

```python
import perth
import librosa

AUDIO_PATH = "YOUR_FILE.wav"

# Load the watermarked audio
watermarked_audio, sr = librosa.load(AUDIO_PATH, sr=None)

# Initialize watermarker (same as used for embedding)
watermarker = perth.PerthImplicitWatermarker()

# Extract watermark
watermark = watermarker.get_watermark(watermarked_audio, sample_rate=sr)
print(f"Extracted watermark: {watermark}")
# Output: 0.0 (no watermark) or 1.0 (watermarked)
```

# Official Discord

👋 Join us on [Discord](https://discord.gg/rJq9cRJBJ6) and let's build something awesome together!

# Citation

If you find this model useful, please consider citing.

```
@misc{chatterboxtts2025,
  author       = {{Resemble AI}},
  title        = {{Chatterbox-TTS}},
  year         = {2025},
  howpublished = {\url{https://github.com/resemble-ai/chatterbox}},
  note         = {GitHub repository}
}
```

# Disclaimer

Don't use this model to do bad things. Prompts are sourced from freely available data on the internet.
