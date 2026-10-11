## This is a light version of chatterbox that removes:
* ```librosa``` and all its required dependencies and instead uses ```torchaudio``` and ```scipy```
* ```perth``` because who the heck needs watermarking anyways
* ```omegaconf``` and instead use standard python
* others I forget, but everything works

However, I did add ```soundfile``` because I like it.

# Installation
>Go through these steps in order.
```
python -m venv .
```

```
.\Scripts\activate
```

```
python.exe -m pip install --upgrade pip
```

```
pip install uv
```

Next, make sure appropriate versions of torch, torchaudio, and CUDA are installed.

```
uv pip install chatterbox-light
```

To install the latest code from GitHub instead:
```
uv pip install -r requirements.txt
```

```
pip install git+https://github.com/BBC-Esq/chatterbox-light.git --no-deps
```

This package installs the `chatterbox` module, just like Resemble AI's `chatterbox-tts`, so don't install both in one environment. Installs of this fork made from GitHub before version 1.0.1 are also named `chatterbox-tts`; run `pip uninstall chatterbox-tts` before installing this.

Optional: if you'll use the multilingual model with Japanese or Chinese text, also install these (or install `"chatterbox-light[multilingual]"`). Without them, Japanese kanji are dropped and Chinese text isn't split into words. The first time Chinese text is used, spacy-pkuseg downloads a ~35 MB model to `~/.pkuseg`.
```
uv pip install pykakasi spacy-pkuseg
```

# Usage
Each model's `generate()` returns a `(1, samples)` tensor at `model.sr` (24 kHz). Models download from Hugging Face the first time they're used. Save the audio with `soundfile`:
```python
import soundfile as sf
sf.write("output.wav", wav.squeeze(0).numpy(), model.sr)
```

### Chatterbox (English)
```python
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")  # or "cpu" / "mps"
wav = model.generate("Hello there, this is Chatterbox.")

# clone a voice from a reference clip
wav = model.generate("Hello there, this is Chatterbox.", audio_prompt_path="reference.wav")

# more expressive: raise exaggeration (default 0.5) and lower cfg_weight (default 0.5) for slower pacing
wav = model.generate("Hello there, this is Chatterbox.", exaggeration=0.7, cfg_weight=0.3)
```

### Chatterbox Turbo (English)
Faster, and understands paralinguistic tags such as `[laugh]`, `[chuckle]` and `[cough]`. A reference clip must be longer than 5 seconds. `cfg_weight`, `exaggeration` and `min_p` are ignored.
```python
from chatterbox.tts_turbo import ChatterboxTurboTTS

model = ChatterboxTurboTTS.from_pretrained(device="cuda")
wav = model.generate("Hi there [chuckle], have you got a minute to chat?", audio_prompt_path="reference.wav")
```

### Chatterbox Nano (English)
A smaller version of Turbo for CPU and tight memory budgets, loaded through the same class with `nano=True`:
```python
from chatterbox.tts_turbo import ChatterboxTurboTTS

model = ChatterboxTurboTTS.from_pretrained(device="cpu", nano=True)
wav = model.generate("Hi there [chuckle], have you got a minute to chat?")
```

### Chatterbox Multilingual (23 languages)
The v2 checkpoint loads by default. Pass `t3_model="v3"` for the newer v3 checkpoint, which upstream recommends (fewer hallucinations, better speaker similarity). For Japanese or Chinese text, install the optional packages above.
```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

print(SUPPORTED_LANGUAGES)  # language codes such as "fr", "de", "ja", "zh"
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda", t3_model="v3")
wav = model.generate("Bonjour, comment ça va ?", language_id="fr")
wav = model.generate("Hola, ¿cómo estás?", language_id="es", audio_prompt_path="reference.wav")
```

### Voice conversion
```python
from chatterbox.vc import ChatterboxVC

model = ChatterboxVC.from_pretrained(device="cuda")
wav = model.generate("input.wav", target_voice_path="target_voice.wav")
```

See the `example_*.py` scripts for complete runs.

Enjoy!
