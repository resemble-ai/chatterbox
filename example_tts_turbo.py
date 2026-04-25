import torchaudio as ta
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Load the Turbo model
model = ChatterboxTurboTTS.from_pretrained(device="cuda")

# Generate with Paralinguistic Tags
text = "Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. It's the SkyNet T-800 series and it's got basically everything. Including AI integration with ChatGPT and all that jazz. Would you like me to get some prices for you?"

# text = "[dispassionately] The mark should exit the building in twelve minutes [suddenly booming] OH MY GOD that puppet bit! [voice tightening with suppressed anger] Stop making me laugh, I have a reputation."

# Generate audio (requires a reference clip for voice cloning)
# wav = model.generate(text, audio_prompt_path="your_10s_ref_clip.wav")
wav = model.generate(text)
ta.save("test-turbo.wav", wav, model.sr)

AUDIO_PROMPT_PATH = "hf_canopy/samples/wavs/039.wav"
if Path(AUDIO_PROMPT_PATH).exists():
    wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
    ta.save("testturbo-inf_dataset.wav", wav, model.sr)
else:
    print(f"Warning: audio prompt file '{AUDIO_PROMPT_PATH}' not found, skipping voice cloning example.")
