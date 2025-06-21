import torch
import torchaudio as ta
from chatterbox.vc import ChatterboxVC

# Monkey patch torch.load to handle device mapping
original_torch_load = torch.load

def patched_torch_load(f, map_location=None, **kwargs):
    """
    Patched torch.load that automatically maps CUDA tensors to CPU/MPS
    """
    if map_location is None:
        # Default to CPU for compatibility
        map_location = 'cpu'
    return original_torch_load(f, map_location=map_location, **kwargs)

torch.load = patched_torch_load

# Automatically detect the best available device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

model = ChatterboxVC.from_pretrained(device=device)

# To convert a file
INPUT_AUDIO_PATH = "YOUR_FILE.wav"
# To use a different target voice
# TARGET_VOICE_PATH = "YOUR_TARGET.wav"

wav = model.generate(
    INPUT_AUDIO_PATH,
    # target_voice_path=TARGET_VOICE_PATH
)
ta.save("test_vc.wav", wav, model.sr)
print("Audio saved to test_vc.wav")
