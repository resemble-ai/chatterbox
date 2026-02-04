
import os
import torch
import soundfile as sf
import numpy as np
from chatterbox.asr import SpeechRecognizer

def verify_stt():
    print("Testing Speech Recognition Module...")
    
    # Initialize Recognizer
    recognizer = SpeechRecognizer(model_id="openai/whisper-tiny")
    
    # Create a dummy audio file (1 second silence)
    dummy_audio_path = "test_audio.wav"
    sr = 16000
    audio_data = np.random.uniform(-0.1, 0.1, sr) # White noise
    sf.write(dummy_audio_path, audio_data, sr)
    
    # Transcribe
    print(f"Transcribing {dummy_audio_path}...")
    text = recognizer.transcribe(dummy_audio_path, language_id="ml")
    
    print(f"Transcription Result: {text}")
    
    # Clean up
    if os.path.exists(dummy_audio_path):
        os.remove(dummy_audio_path)

if __name__ == "__main__":
    verify_stt()
