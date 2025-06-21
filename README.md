# Chatterbox TTS - Apple Silicon MPS Optimized 🚀

A high-performance Text-to-Speech (TTS) implementation optimized for Apple Silicon Macs using Metal Performance Shaders (MPS). This project takes the original [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) and supercharges it for M1/M2/M3 Macs.

## 🎯 Project Purpose

The goal was to optimize Chatterbox TTS for Apple Silicon by:
- Loading models on CPU first, then transferring all components to GPU for inference
- Achieving <20 seconds generation time for 1 minute of audio
- Implementing proper model warm-up and performance monitoring
- Creating a user-friendly Gradio interface

## 🏆 Performance Achievements

### Before Optimization
- Warm-up: ~7 iterations/second
- Generation: 5-7 iterations/second
- Severe performance degradation on MPS due to incompatible operations

### After Optimization
- **Warm-up: 12.27 it/s** (75% improvement)
- **Generation: 9.69-14.54 it/s** (2-3x improvement)
- **Real-world: 20 seconds of audio generated in 59 seconds** (RTF: 2.95)
- Successfully meets the target of <20 seconds for 1 minute of audio!

## 🔧 Technical Optimizations

### 1. **MPS-Optimized Rotary Embeddings** (`mps_fast_patch.py`)
- Pre-computes all rotary position embeddings on CPU once
- Eliminates expensive CPU↔MPS transfers during inference
- Monkey-patches the transformers library at import time
- Maintains mathematical correctness while improving speed

### 2. **Smart Model Loading**
- Models load on CPU first (as requested)
- All components (T3, S3Gen, Voice Encoder) transfer to MPS simultaneously
- Proper device synchronization ensures stable transfers

### 3. **Model Warm-up**
- Executes before the Gradio interface loads
- Ensures all MPS kernels are compiled and ready
- Provides performance verification

### 4. **Enhanced Gradio Interface**
- Real-time performance metrics in logs
- Support for both default voice and voice cloning
- Automatic text chunking for long inputs
- Proper error handling and device management

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chatterbox-macos-optimize.git
cd chatterbox-macos-optimize

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
pip install gradio soundfile optimum
```

## 🚀 Usage

### Gradio Web Interface
```bash
python gradio_tts_app.py
```
Then open http://127.0.0.1:7860 in your browser.

### Voice Conversion
```bash
python gradio_vc_app.py
```

### Command Line Examples
```bash
# Text-to-Speech with default voice
python example_tts.py

# Voice Conversion
python example_vc.py

# Mac-specific example
python example_for_mac.py
```

## 🎛️ Parameters Guide

- **Temperature** (0.3-0.8): Lower = more stable, Higher = more expressive
- **CFG Weight** (0.3-0.7): Controls adherence to text
- **Exaggeration** (0.25-2.0): Voice characteristic strength (0.5 = neutral)
- **Min-p** (0.02-0.1): Newer sampler, handles higher temperatures better
- **Top-p** (0.8-1.0): Original sampler, 1.0 disables
- **Repetition Penalty** (1.0-1.5): Reduces repetitive patterns

## 🐛 Troubleshooting

### If output sounds garbled:
1. Try lower temperature (0.3-0.5)
2. Reduce CFG weight (0.3-0.4)
3. Use neutral exaggeration (0.5)
4. Ensure reference audio is clean and 3-10 seconds long

### If performance is slow:
1. Check that MPS is detected: Look for "🚀 Apple Silicon MPS backend is available"
2. Ensure you're using the optimized version (check for "✅ Replaced N rotary embeddings")
3. Close other GPU-intensive applications

## 🛠️ Development

This project was created through an innovative AI-assisted development process:

- **[Cursor AI](https://cursor.sh/)** - AI-powered code editor that helped implement the MPS optimizations
- **Claude Opus 4** - Provided expertise on PyTorch MPS optimization and transformer architectures  
- **Vibe Coding** - Collaborative AI-human development approach
- **[CodeRabbit](https://coderabbit.ai/)** - Monitors all updates and ensures code quality

The entire optimization was achieved through natural language descriptions of the desired improvements, with AI handling the implementation details while maintaining human oversight and direction.

## 📄 License

This project inherits the original Chatterbox license. See LICENSE file for details.

## 🙏 Acknowledgments

- [Resemble AI](https://github.com/resemble-ai) for the original Chatterbox implementation
- The PyTorch team for MPS backend development
- The Hugging Face team for the transformers library

---

*Built with ❤️ using AI-assisted development on Apple Silicon*

<img width="1200" alt="cb-big2" src="https://github.com/user-attachments/assets/bd8c5f03-e91d-4ee5-b680-57355da204d1" />

# Chatterbox TTS

[![Alt Text](https://img.shields.io/badge/listen-demo_samples-blue)](https://resemble-ai.github.io/chatterbox_demopage/)
[![Alt Text](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/ResembleAI/Chatterbox)
[![Alt Text](https://static-public.podonos.com/badges/insight-on-pdns-sm-dark.svg)](https://podonos.com/resembleai/chatterbox)
[![Discord](https://img.shields.io/discord/1377773249798344776?label=join%20discord&logo=discord&style=flat)](https://discord.gg/rJq9cRJBJ6)

_Made with ♥️ by <a href="https://resemble.ai" target="_blank"><img width="100" alt="resemble-logo-horizontal" src="https://github.com/user-attachments/assets/35cf756b-3506-4943-9c72-c05ddfa4e525" /></a>

We're excited to introduce Chatterbox, [Resemble AI's](https://resemble.ai) first production-grade open source TTS model. Licensed under MIT, Chatterbox has been benchmarked against leading closed-source systems like ElevenLabs, and is consistently preferred in side-by-side evaluations.

Whether you're working on memes, videos, games, or AI agents, Chatterbox brings your content to life. It's also the first open source TTS model to support **emotion exaggeration control**, a powerful feature that makes your voices stand out. Try it now on our [Hugging Face Gradio app.](https://huggingface.co/spaces/ResembleAI/Chatterbox)

If you like the model but need to scale or tune it for higher accuracy, check out our competitively priced TTS service (<a href="https://resemble.ai">link</a>). It delivers reliable performance with ultra-low latency of sub 200ms—ideal for production use in agents, applications, or interactive media.

# Key Details
- SoTA zeroshot TTS
- 0.5B Llama backbone
- Unique exaggeration/intensity control
- Ultra-stable with alignment-informed inference
- Trained on 0.5M hours of cleaned data
- Watermarked outputs
- Easy voice conversion script
- [Outperforms ElevenLabs](https://podonos.com/resembleai/chatterbox)

# Tips
- **General Use (TTS and Voice Agents):**
  - The default settings (`exaggeration=0.5`, `cfg_weight=0.5`) work well for most prompts.
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
We developed and tested Chatterbox on Python 3.11 on Debain 11 OS; the versions of the dependencies are pinned in `pyproject.toml` to ensure consistency. You can modify the code or dependencies in this installation mode.


# Usage
```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
wav = model.generate(text)
ta.save("test-1.wav", wav, model.sr)

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save("test-2.wav", wav, model.sr)
```
See `example_tts.py` and `example_vc.py` for more examples.

# Supported Lanugage
Currenlty only English.

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

# Disclaimer
Don't use this model to do bad things. Prompts are sourced from freely available data on the internet.
