"""
uv run examples/example_for_mac.py
"""
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from chatterbox.models.utils import get_device

# Detect device (Mac with M1/M2/M3/M4)
device = get_device()
map_location = torch.device(device)

model = ChatterboxTTS.from_pretrained(device=device)
text = "Today is the day. I want to move like a titan at dawn, sweat like a god forging lightning. No more excuses. From now on, my mornings will be temples of discipline. I am going to work out like the gods… every damn day."

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
wav = model.generate(
    text, 
    audio_prompt_path=AUDIO_PROMPT_PATH,
    exaggeration=2.0,
    cfg_weight=0.5
    )
ta.save("test-2.wav", wav, model.sr)
