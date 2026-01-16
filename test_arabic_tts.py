#!/usr/bin/env python3
"""
Test Arabic TTS capabilities of Chatterbox Multilingual model.
"""
import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    device = get_device()
    print(f"🚀 Using device: {device}")
    
    print("📥 Loading Chatterbox Multilingual model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    print("✅ Model loaded successfully!")
    
    # Test Arabic texts - Modern Standard Arabic and common phrases
    arabic_texts = [
        ("greeting", "مرحباً، كيف حالك اليوم؟ أتمنى لك يوماً سعيداً."),
        ("news_style", "في الشهر الماضي، وصلنا إلى معلم جديد بمليارين من المشاهدات على قناتنا على يوتيوب."),
        ("formal", "نرحب بكم في هذا البرنامج، ونتمنى لكم مشاهدة ممتعة."),
    ]
    
    print("\n🎤 Generating Arabic speech samples...")
    
    for name, text in arabic_texts:
        print(f"\n📝 Text ({name}): {text}")
        
        # Generate with default settings
        wav = model.generate(
            text,
            language_id="ar",
            exaggeration=0.5,
            cfg_weight=0.5,
            temperature=0.8,
        )
        
        output_path = f"arabic_output_{name}.wav"
        ta.save(output_path, wav, model.sr)
        print(f"💾 Saved: {output_path}")
    
    print("\n✅ All Arabic samples generated successfully!")
    print("🎧 Listen to the generated .wav files to evaluate quality.")

if __name__ == "__main__":
    main()

