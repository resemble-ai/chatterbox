import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

model = ChatterboxTTS.from_pretrained(device=device)

text = ["Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill.",
        "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill.",
        "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill.",
        "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."]
wavs = model.generate(text)

# Save each generated waveform to a separate file
for i, wav in enumerate(wavs):
    ta.save(f"test-batch-{i+1}.wav", wav, model.sr)


# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "C:\\_myDrive\\repos\\auto-vlog\\assets\\audio_sample1.wav"
try:
    wavs = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
    for i, wav in enumerate(wavs):
        ta.save(f"test-prompted-batch-{i+1}.wav", wav, model.sr)
except FileNotFoundError:
    print(f"\nSkipping second example because audio prompt not found at: '{AUDIO_PROMPT_PATH}'")
    print("Please replace 'YOUR_FILE.wav' with a real audio file path to test this.")
