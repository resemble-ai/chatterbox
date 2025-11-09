
<img width="1200" height="600" alt="Chatterbox-Multilingual" src="https://www.resemble.ai/wp-content/uploads/2025/09/Chatterbox-Multilingual-1.png" />

# Chatterbox TTS

[![Alt Text](https://img.shields.io/badge/listen-demo_samples-blue)](https://resemble-ai.github.io/chatterbox_demopage/)
[![Alt Text](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/ResembleAI/Chatterbox)
[![Alt Text](https://static-public.podonos.com/badges/insight-on-pdns-sm-dark.svg)](https://podonos.com/resembleai/chatterbox)
[![Discord](https://img.shields.io/discord/1377773249798344776?label=join%20discord&logo=discord&style=flat)](https://discord.gg/rJq9cRJBJ6)

_Made with ♥️ by <a href="https://resemble.ai" target="_blank"><img width="100" alt="resemble-logo-horizontal" src="https://github.com/user-attachments/assets/35cf756b-3506-4943-9c72-c05ddfa4e525" /></a>

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
wav_french = multilingual_model.generate(french_text, language_id="fr")
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

# Fine-tuning Guide

This repository includes tools for fine-tuning Chatterbox Multilingual TTS using LoRA (Low-Rank Adaptation). Follow this guide to train the model on your custom dataset.

## Overview

The fine-tuning process involves two main scripts:
- **`lora.py`** - Main training script with LoRA fine-tuning
- **`fix_merged_model.py`** - Converts the trained model to the correct format

## Step 1: Prepare Your Dataset

Create the following directory structure:
```
audio_data/
├── metadata.csv
└── audio/
    ├── utterance_0001.wav
    ├── utterance_0002.wav
    └── ...
```

**metadata.csv format:**
```csv
file_name,transcription,duration_seconds
audio/utterance_0001.wav,Your transcription text here,3.45
audio/utterance_0002.wav,Another transcription,2.87
```

**Required columns:**
- `file_name` - Relative path to audio file (e.g., `audio/utterance_0001.wav`)
- `transcription` - Text transcription of the audio
- `duration_seconds` - (Optional) Duration in seconds for faster loading

**Audio requirements:**
- Format: WAV files
- Duration: Between 1-400 seconds (configurable)
- Sample rate: Any (will be resampled automatically)
- Quality: Clean speech with accurate transcriptions

## Step 2: Configure Training Parameters

Edit the configuration section at the top of `lora.py`:

```python
# Data and paths
AUDIO_DATA_DIR = "./audio_data"           # Path to your dataset
CHECKPOINT_DIR = "checkpoints_lora"       # Where to save checkpoints

# Training hyperparameters
BATCH_SIZE = 1                            # Batch size (1 for most GPUs)
EPOCHS = 50                               # Number of training epochs
LEARNING_RATE = 2e-5                      # Learning rate
GRADIENT_ACCUMULATION_STEPS = 8           # Accumulate gradients over N steps

# LoRA parameters
LORA_RANK = 32                            # LoRA rank (lower = fewer parameters)
LORA_ALPHA = 64                           # LoRA alpha (scaling factor)
LORA_DROPOUT = 0.05                       # Dropout rate

# Audio constraints
MAX_AUDIO_LENGTH = 400.0                  # Max audio length in seconds
MIN_AUDIO_LENGTH = 1.0                    # Min audio length in seconds
MAX_TEXT_LENGTH = 1000                    # Max text length in characters

# Checkpointing
SAVE_EVERY_N_STEPS = 200                  # Save checkpoint every N steps
VALIDATION_SPLIT = 0.1                    # 10% of data for validation
```

**Language Configuration:**
By default, the script trains on Arabic (`language_id='ar'`). To change the language, edit line 1079 in `lora.py`:
```python
language_id='ar'  # Change to: 'en', 'fr', 'zh', 'es', etc.
```

## Step 3: Run Training

Start the training process:

```bash
python lora.py
```

**What happens during training:**
1. Loads the Chatterbox Multilingual TTS model
2. Injects LoRA adapters into transformer layers
3. Trains only the LoRA parameters (efficient fine-tuning)
4. Saves checkpoints every 200 steps
5. Generates real-time training metrics visualization (`training_metrics.png`)
6. Creates a merged model at the end

**Training outputs:**
- `checkpoints_lora/checkpoint_epochX_stepY.pt` - Training checkpoints
- `checkpoints_lora/final_lora_adapter.pt` - Final LoRA weights
- `checkpoints_lora/merged_model/` - Merged model (base + LoRA)
- `training_metrics.png` - Real-time training visualization

**Training metrics:**
The script generates a live dashboard showing:
- Training and validation loss
- Learning rate schedule
- Gradient norms
- Recent batch losses
- Loss variance
- Time per training step

**GPU requirements:**
- Minimum: 16GB VRAM (NVIDIA GPU)
- Recommended: 24GB+ VRAM for faster training
- CPU training is supported but significantly slower

## Step 4: Convert the Merged Model

After training completes, convert the model to the correct format:

```bash
python fix_merged_model.py
```

This converts the PyTorch `.pt` files to `.safetensors` format required by `ChatterboxMultilingualTTS.from_local()`.

**Output:**
```
checkpoints_lora/merged_model/
├── ve.pt
├── t3_mtl23ls_v2.pt
├── t3_mtl23ls_v2.safetensors  ← Created by fix_merged_model.py
├── s3gen.pt
├── grapheme_mtl_merged_expanded_v1.json
└── conds.pt
```

## Step 5: Test Your Fine-tuned Model

### Option A: Load the Merged Model

```python
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Load your fine-tuned model
model = ChatterboxMultilingualTTS.from_local(
    "./checkpoints_lora/merged_model",
    device="cuda"
)

# Generate speech with your fine-tuned voice
text = "مرحبا، هذا نموذج الصوت المخصص الخاص بي"  # Arabic example
wav = model.generate(text, language_id="ar")
ta.save("finetuned_output.wav", wav, model.sr)
```

### Option B: Load Base Model + LoRA Adapter

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from lora import load_lora_adapter

# Load base model
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

# Load your LoRA adapter
lora_layers = load_lora_adapter(
    model,
    "./checkpoints_lora/final_lora_adapter.pt",
    device="cuda"
)

# Generate speech
text = "Your text here"
wav = model.generate(text, language_id="ar")
```

## Troubleshooting

### Common Issues

**"No valid audio samples found"**
- Check that `AUDIO_DATA_DIR` in `lora.py` matches your dataset location
- Verify `metadata.csv` exists and has the correct format
- Ensure audio files are in the `audio/` subdirectory

**"CUDA out of memory"**
- Reduce `BATCH_SIZE` to 1
- Reduce `MAX_AUDIO_LENGTH` to 200 or less
- Reduce `LORA_RANK` to 16 or 8
- Use gradient checkpointing (advanced)

**"Loss is NaN or not decreasing"**
- Lower `LEARNING_RATE` (try 1e-5)
- Check that transcriptions match audio content
- Ensure audio quality is good (no noise/corruption)
- Increase `WARMUP_STEPS` to 1000

**"Training is very slow"**
- Reduce `MAX_AUDIO_LENGTH` to filter long samples
- Use a GPU instead of CPU
- Increase `GRADIENT_ACCUMULATION_STEPS` and `BATCH_SIZE`

### Dataset Quality Tips

For best results:
- **Accurate transcriptions** - Ensure text exactly matches spoken audio
- **Clean audio** - Remove background noise, music, and overlapping speech
- **Consistent speaker** - Use recordings from the same speaker
- **Sufficient data** - Aim for at least 30-60 minutes of audio
- **Diverse content** - Include varied vocabulary and sentence structures

## Advanced Configuration

### Target Modules

LoRA is applied to these transformer layers (line 745 in `lora.py`):
```python
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
```

To fine-tune fewer layers (faster, less overfitting):
```python
target_modules = ["q_proj", "v_proj"]  # Only query and value projections
```

### Resume Training from Checkpoint

To resume training from a checkpoint, modify line 741 in `lora.py`:
```python
# Replace:
model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)

# With:
model = ChatterboxMultilingualTTS.from_local(
    "./checkpoints_lora/merged_model",
    device=DEVICE
)
```

### Multi-Language Fine-tuning

To train on multiple languages, modify the `load_audio_samples()` function to read language IDs from `metadata.csv`:

1. Add a `language_id` column to `metadata.csv`:
```csv
file_name,transcription,duration_seconds,language_id
audio/file1.wav,Hello world,2.5,en
audio/file2.wav,Bonjour monde,2.3,fr
```

2. Update line 1079 in `lora.py`:
```python
# Replace:
language_id='ar'

# With:
language_id=row.get('language_id', 'ar')  # Read from metadata
```

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
