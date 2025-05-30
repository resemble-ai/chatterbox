import torch
import torchaudio as ta

from chatterbox.vc import ChatterboxVC

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

model = ChatterboxVC.from_pretrained(device)
wav = model.generate(
    audio="test/male_conan.mp3",
    target_voice_path="test/male_petergriffin.wav",
)
ta.save("testvc.wav", wav, model.sr)
