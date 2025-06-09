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

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
result = model.generate(text)
audio_tensor = result['audio_tensor']
timestamps = result['timestamps']
ta.save("test-1.wav", audio_tensor, model.sr)

print("\nTimestamps for the first example:")
if timestamps:
    for i, ts_entry in enumerate(timestamps[:5]): # Print first 5 timestamps
        print(f"  Token {i+1}: '{ts_entry['token_text']}', Start: {ts_entry['start_time']:.2f}s, End: {ts_entry['end_time']:.2f}s")
    if len(timestamps) > 5:
        print(f"  ... and {len(timestamps) - 5} more timestamp entries.")
else:
    print("  No timestamps generated.")

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save("test-2.wav", wav, model.sr)
