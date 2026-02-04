
import os
import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

def verify_malayalam():
    # Automatically detect the best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Load model
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    
    # Test Malayalam
    text = "കഴിഞ്ഞ മാസം, ഞങ്ങളുടെ YouTube ചാനലിൽ രണ്ട് ബില്യൺ കാഴ്‌ചകൾ എന്ന പുതിയ നാഴികക്കല്ല് ഞങ്ങൾ പിന്നിട്ടു."
    lang_id = "ml"
    
    print(f"Generating audio for language: {lang_id}")
    try:
        wav = model.generate(text, language_id=lang_id)
        output_path = "verify_ml.wav"
        ta.save(output_path, wav, model.sr)
        print(f"Success! Saved to {output_path}")
    except Exception as e:
        print(f"Failed to generate Malayalam audio: {e}")

if __name__ == "__main__":
    verify_malayalam()
