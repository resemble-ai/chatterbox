#!/usr/bin/env python3
"""
Example script demonstrating multilingual TTS functionality.
This shows how to use the new ChatterboxMultilingualTTS class.
"""

import torchaudio as ta
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

def main():
    # Automatically detect the best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    
    # Get supported languages
    supported_languages = ChatterboxMultilingualTTS.get_supported_languages()
    print(f"Supported languages ({len(supported_languages)}):")
    for code, name in supported_languages.items():
        print(f"  {code}: {name}")
    
    # Note: This example shows how to use the multilingual TTS
    # The actual model loading would require the multilingual model files
    
    print("\nExample usage (requires multilingual model files):")
    print("# Load multilingual model")
    print("model = ChatterboxMultilingualTTS.from_pretrained(device=device)")
    print()
    
    # Example texts in different languages
    example_texts = {
        "en": "Hello world, this is a test of multilingual text-to-speech synthesis.",
        "es": "Hola mundo, esta es una prueba de síntesis de texto a voz multilingüe.",
        "fr": "Bonjour le monde, ceci est un test de synthèse vocale multilingue.",
        "de": "Hallo Welt, das ist ein Test der mehrsprachigen Text-zu-Sprache-Synthese.",
        "ja": "こんにちは世界、これは多言語音声合成のテストです。",
        "zh": "你好世界，这是多语言文本转语音合成的测试。",
        "ko": "안녕하세요 세계, 이것은 다국어 텍스트 음성 변환 합성 테스트입니다.",
        "ar": "مرحبا بالعالم، هذا اختبار لتوليف النص إلى كلام متعدد اللغات.",
        "he": "שלום עולם, זה מבחן של סינתזה של טקסט לדיבור רב-לשוני.",
    }
    
    print("Example text generation for different languages:")
    for lang_code, text in example_texts.items():
        if lang_code in supported_languages:
            print(f"\n# Generate {supported_languages[lang_code]} ({lang_code}) audio:")
            print(f"text = \"{text}\"")
            print(f"wav = model.generate(text, language_id='{lang_code}')")
            print(f"ta.save('output_{lang_code}.wav', wav, model.sr)")
    
    print("\n# Using custom audio prompt for voice cloning:")
    print("AUDIO_PROMPT_PATH = 'your_reference_voice.wav'")
    print("wav = model.generate(text, language_id='en', audio_prompt_path=AUDIO_PROMPT_PATH)")
    print("ta.save('cloned_voice.wav', wav, model.sr)")
    
    print("\n# Advanced parameters:")
    print("wav = model.generate(")
    print("    text='Hello world',")
    print("    language_id='en',")
    print("    temperature=0.8,      # Controls randomness")
    print("    cfg_weight=0.5,       # Classifier-free guidance")
    print("    exaggeration=0.5,     # Emotion control")
    print("    min_p=0.05,           # Min-P sampling")
    print("    top_p=1.0,            # Top-P sampling")
    print("    repetition_penalty=1.2 # Reduce repetition")
    print(")")
    
    print("\nNote: To actually run this example, you need:")
    print("1. Multilingual model files from Hugging Face")
    print("2. A reference audio file for voice cloning")
    print("3. Sufficient GPU memory for the models")

if __name__ == "__main__":
    main()
