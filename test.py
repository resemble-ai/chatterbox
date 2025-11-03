#!/usr/bin/env python3
"""
Simple test script for Arabic TTS - based on official Chatterbox example
"""

import torchaudio as ta
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import os

def main():
    print("=" * 60)
    print("Simple Arabic TTS Test")
    print("=" * 60)
    print()
    
    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    print()
    
    # Check if merged model exists
    model_path = "/teamspace/studios/this_studio/chatterbox/checkpoints_lora/merged_model"
    
    if os.path.exists(model_path):
        print(f"📂 Loading fine-tuned model from: {model_path}")
        try:
            # Try loading from local path
            model = ChatterboxMultilingualTTS.from_local(model_path, device=device)
            print("✅ Fine-tuned model loaded successfully!")
        except Exception as e:
            print(f"⚠️  Could not load fine-tuned model: {e}")
            print("\n📥 Loading pretrained model from HuggingFace instead...")
            model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            print("✅ Pretrained model loaded successfully!")
    else:
        print("⚠️  Fine-tuned model not found")
        print(f"📥 Loading pretrained model from HuggingFace...")
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        print("✅ Pretrained model loaded successfully!")
    
    print()
    
    # Create output directory
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # Test sentences
    test_sentences = [
        ("Egyptian Arabic", "فروع البنك فاتحة من الساعة تمانية ونص الصبح لحد تلاتة بعد الضهر، ما عدا الجمعة والسبت أجازة."),
        ("Gulf Arabic", "يا جماعة، الطريق الدائري فيه شوية زحمة ناحية المعادي، ياريت تاخدوا بالكم و تحاولوا تشوفوا طرق بديلة."),
        ("MSA", "شنو حالك اليوم تري؟ باجر نراك ان شاء الله"),
        ("Arabic greeting", "مرحبا بك في هذا البرنامج"),
    ]
    
    print("=" * 60)
    print("Generating audio...")
    print("=" * 60)
    print()
    
    for idx, (label, text) in enumerate(test_sentences):
        filename = f"{output_dir}/test_{idx + 1:02d}.wav"
        
        print(f"{idx + 1}. {label}")
        print(f"   Text: {text}")
        
        try:
            # Generate audio with language_id="ar" for Arabic
            wav = model.generate(text, language_id="ar")
            ta.save(filename, wav, model.sr)
            
            # Get file info
            size_kb = os.path.getsize(filename) / 1024
            duration = wav.shape[-1] / model.sr
            
            print(f"   ✅ Saved: {filename} ({size_kb:.1f} KB, {duration:.2f}s)")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()
    
    print("=" * 60)
    print("✅ Done!")
    print("=" * 60)
    print()
    print(f"📂 Audio files saved in: {output_dir}/")
    print()
    print("🎧 To play:")
    print(f"   ffplay {output_dir}/test_01.wav")
    print()

if __name__ == "__main__":
    main()

