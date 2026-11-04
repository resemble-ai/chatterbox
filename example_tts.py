import soundfile as sf
import torch
from pathlib import Path
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
sf.write("test-1.wav", wav.squeeze(0).numpy(), model.sr, subtype="FLOAT")

multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
# v2 is the default. Pass t3_model="v3" to use the v3 multilingual checkpoint.
text = "Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox, il prend en charge 23 langues."
wav = multilingual_model.generate(text, language_id="fr")
sf.write("test-2.wav", wav.squeeze(0).numpy(), multilingual_model.sr, subtype="FLOAT")


# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
if Path(AUDIO_PROMPT_PATH).exists():
    wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
    sf.write("test-3.wav", wav.squeeze(0).numpy(), model.sr, subtype="FLOAT")
else:
    print(f"Warning: audio prompt file '{AUDIO_PROMPT_PATH}' not found, skipping voice cloning example.")
