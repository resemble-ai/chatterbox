import torch
try:
    import numpy as np
except Exception:
    np = None

# Use soundfile for WAV saving instead of torchaudio. We import it softly so
# importing this example file won't fail if the user doesn't have soundfile
# installed; a clear runtime error will be raised when saving is attempted.
try:
    import soundfile as sf
except Exception:
    sf = None

from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

model = ChatterboxTTS.from_pretrained(device=device)

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
wav = model.generate(text)
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

save_wav("test-1.wav", wav, model.sr)

multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
text = "Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox, il prend en charge 23 langues."
wav = multilingual_model.generate(text, language_id="fr")
save_wav("test-2.wav", wav, multilingual_model.sr)


# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
save_wav("test-3.wav", wav, model.sr)
