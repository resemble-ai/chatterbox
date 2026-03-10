import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from chatterbox.vc import ChatterboxVC

# Detect device (Mac with M1/M2/M3/M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Example 1: Basic TTS with default voice
print("\nExample 1: Basic TTS with default voice")
model_tts = ChatterboxTTS.from_pretrained(device=device)

text = "Today is the day. I want to move like a titan at dawn, sweat like a god forging lightning. No more excuses. From now on, my mornings will be temples of discipline. I am going to work out like the gods… every damn day."
wav = model_tts.generate(text)
ta.save("example1_default_voice.wav", wav, model_tts.sr)

# Example 2: TTS with custom voice and emotion
print("\nExample 2: TTS with custom voice and emotion")
# If you want to synthesize with a different voice, specify the audio prompt
# AUDIO_PROMPT_PATH = "YOUR_FILE.wav"  # Uncomment and set your audio file path
# wav = model_tts.generate(
#     text, 
#     audio_prompt_path=AUDIO_PROMPT_PATH,
#     exaggeration=2.0,  # Higher emotion
#     cfg_weight=0.5     # Default pacing
# )
# ta.save("example2_custom_voice.wav", wav, model_tts.sr)

# Example 3: Voice Conversion
print("\nExample 3: Voice Conversion")
model_vc = ChatterboxVC.from_pretrained(device=device)

# To convert a voice, you need:
# 1. Source audio to convert
# SOURCE_AUDIO = "YOUR_SOURCE.wav"  # Uncomment and set your source audio path
# 2. Target voice to convert to
# TARGET_VOICE = "YOUR_TARGET.wav"  # Uncomment and set your target voice path
# wav = model_vc.generate(SOURCE_AUDIO, target_voice_path=TARGET_VOICE)
# ta.save("example3_voice_conversion.wav", wav, model_vc.sr)

print("\nExamples completed. Check the generated WAV files.")
print("Note: Examples 2 and 3 are commented out. Uncomment and set your audio file paths to try them.")
