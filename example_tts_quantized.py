"""
Example demonstrating BNB (bitsandbytes) quantization for memory-efficient inference.

BNB quantization reduces memory usage by ~50% by converting model weights to 8-bit integers.
This is particularly useful for running models on GPUs with limited VRAM.

Requirements:
- CUDA-compatible GPU
- bitsandbytes library: pip install bitsandbytes

Note: BNB quantization is only supported on CUDA devices.
"""

import torch
try:
    import numpy as np
except Exception:
    np = None

# Use soundfile for WAV saving instead of torchaudio
try:
    import soundfile as sf
except Exception:
    sf = None

from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# BNB quantization requires CUDA
if not torch.cuda.is_available():
    raise RuntimeError(
        "BNB quantization requires a CUDA-compatible GPU. "
        "Please use example_tts.py for CPU/MPS inference."
    )

device = "cuda"
print(f"Using device: {device}")

# Load models with BNB quantization enabled
# This will reduce memory usage by approximately 50%
print("\n🔧 Loading ChatterboxTTS with BNB quantization...")
model = ChatterboxTTS.from_pretrained(device=device, use_bnb_quantization=True)
print("✅ ChatterboxTTS loaded successfully")

print("\n🔧 Loading ChatterboxMultilingualTTS with BNB quantization...")
multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device, use_bnb_quantization=True)
print("✅ ChatterboxMultilingualTTS loaded successfully")

def save_wav(path, wav, sr):
    """Save a waveform tensor/array to path using soundfile.

    Accepts torch.Tensor (1D or 2D channels-first) or numpy arrays.
    If soundfile or numpy is not installed, raises a clear RuntimeError.
    """
    if sf is None or np is None:
        raise RuntimeError(
            "Saving WAV requires numpy and soundfile (pysoundfile). "
            "Install with: pip install numpy soundfile"
        )

    # Convert torch Tensor to numpy
    if isinstance(wav, torch.Tensor):
        arr = wav.detach().cpu().numpy()
    else:
        arr = np.asarray(wav)

    # torchaudio uses (channels, samples). soundfile expects (samples, channels)
    if arr.ndim == 2 and arr.shape[0] <= 2:
        arr = arr.T

    # Ensure float32 for soundfile
    arr = arr.astype("float32")
    sf.write(path, arr, sr)

# Generate English speech
print("\n🎙️ Generating English speech...")
text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
wav = model.generate(text)
save_wav("test-quantized-1.wav", wav, model.sr)
print("✅ Saved to test-quantized-1.wav")

# Generate multilingual speech (French)
print("\n🎙️ Generating French speech...")
text = "Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox, il prend en charge 23 langues."
wav = multilingual_model.generate(text, language_id="fr")
save_wav("test-quantized-2.wav", wav, multilingual_model.sr)
print("✅ Saved to test-quantized-2.wav")

# If you want to synthesize with a different voice, specify the audio prompt
# AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
# wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
# save_wav("test-quantized-3.wav", wav, model.sr)

print("\n✨ All done! The quantized models use approximately 50% less GPU memory.")
print("💡 Memory savings are especially noticeable with larger batch sizes or longer sequences.")
