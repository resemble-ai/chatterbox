"""
uv run examples/example_vc.py
"""
import torchaudio as ta

from chatterbox.vc import ChatterboxVC
from chatterbox.models.utils import get_device

# Automatically detect the best available device
device = get_device()

print(f"Using device: {device}")

AUDIO_PATH = "YOUR_FILE.wav"
TARGET_VOICE_PATH = "YOUR_FILE.wav"

model = ChatterboxVC.from_pretrained(device)
wav = model.generate(
    audio=AUDIO_PATH,
    target_voice_path=TARGET_VOICE_PATH,
)
ta.save("testvc.wav", wav, model.sr)
