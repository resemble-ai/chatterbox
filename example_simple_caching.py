"""
Simple example using the ConditionalCache utility.

This demonstrates the easiest way to implement caching for your TTS workflow.
"""

import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from conditional_cache import ConditionalCache

# Setup device
device = "mps" if torch.backends.mps.is_available() else "cpu"
map_location = torch.device(device)

# Patch torch.load
torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)
torch.load = patched_torch_load

print("Loading model...")
model = ChatterboxMultilingualTTS.from_pretrained(device=device)
print("Model loaded!\n")

# Initialize cache (will create ./tts_cache directory)
cache = ConditionalCache(cache_dir="./tts_cache", auto_save=True)

# Your reference voice
VOICE = "julia-whelan.wav"
EXAGGERATION = 0.2

# Multiple texts to generate with the same voice
texts = [
    "It was a dark and stormy night when everything changed.",
    "The rain poured down in sheets, and thunder rumbled in the distance.",
    "Sarah stood at the window, watching the storm rage outside.",
    "She knew that by morning, nothing would ever be the same again.",
]

print("=" * 80)
print("Generating audio with caching enabled")
print("=" * 80)
print(f"Voice: {VOICE}")
print(f"Exaggeration: {EXAGGERATION}")
print(f"Texts to generate: {len(texts)}\n")

# Generate all texts - caching happens automatically!
for i, text in enumerate(texts, 1):
    print(f"[{i}/{len(texts)}] {text}")
    
    # This line does the magic:
    # - First call: prepares conditionals and caches them
    # - Subsequent calls: reuses cached conditionals instantly
    cache.get_or_prepare(model, VOICE, exaggeration=EXAGGERATION)
    
    # Generate audio (fast because conditionals are already prepared)
    wav = model.generate(
        text,
        language_id="en",
        cfg_weight=0.5,
        exaggeration=EXAGGERATION,  # Must match cache.get_or_prepare()
    )
    
    # Save output
    output_file = f"cached_output_{i}.wav"
    ta.save(output_file, wav, model.sr)
    print(f"    Saved: {output_file}\n")

# Show cache statistics
print("=" * 80)
print("Cache Statistics")
print("=" * 80)
stats = cache.get_cache_stats()
for key, value in stats.items():
    print(f"{key}: {value}")

print(f"\n{cache}")
print("\n✅ Done! All audio files generated with efficient caching.")

# Example: Using multiple voices with caching
print("\n" + "=" * 80)
print("Bonus: Multiple voices example")
print("=" * 80)

# You can switch between different voices seamlessly
voices = {
    "julia": ("julia-whelan.wav", 0.2),
    "earl": ("earl-nightingale.wav", 0.5),
}

for voice_name, (voice_file, exag) in voices.items():
    # Check if voice file exists
    import os
    if not os.path.exists(voice_file):
        print(f"⚠ Skipping {voice_name} - file not found: {voice_file}")
        continue
    
    print(f"\nGenerating with {voice_name} voice...")
    
    # Prepare conditionals for this voice (cached automatically)
    cache.get_or_prepare(model, voice_file, exaggeration=exag)
    
    # Generate
    wav = model.generate(
        f"Hello, this is the {voice_name} voice speaking.",
        language_id="en",
        cfg_weight=0.5,
        exaggeration=exag,
    )
    
    ta.save(f"voice_{voice_name}.wav", wav, model.sr)
    print(f"    Saved: voice_{voice_name}.wav")

print(f"\n{cache}")
print("\n" + "=" * 80)
print("Cache contains conditionals for all voices used!")
print("Next time you run this, it will load from disk cache instantly.")
print("=" * 80)
