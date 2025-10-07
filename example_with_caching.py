"""
Example demonstrating conditional caching for efficient TTS generation.

This shows how to cache conditionals (derived from reference audio) and reuse them
for generating multiple different texts, significantly improving performance.
"""

import torch
import torchaudio as ta
import time
from pathlib import Path
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Detect device (Mac with M1/M2/M3/M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Patch torch.load to use correct device
map_location = torch.device(device)
torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)
torch.load = patched_torch_load

print("Loading model...")
model = ChatterboxMultilingualTTS.from_pretrained(device=device)
print("Model loaded!\n")

# Reference audio file
AUDIO_PROMPT_PATH = "julia-whelan.wav"

# Multiple texts to generate with the same voice
texts = [
    "Hello, this is the first sentence.",
    "Now we're generating a second piece of audio with the same voice.",
    "And here's a third one, all using cached conditionals for better performance.",
    "The quick brown fox jumps over the lazy dog.",
]

# Generation parameters
exaggeration = 0.2
cfg_weight = 0.5
language_id = "en"


print("=" * 80)
print("METHOD 1: WITHOUT CACHING (Inefficient)")
print("=" * 80)
print("Calling prepare_conditionals() for each text...\n")

start_time = time.time()
for i, text in enumerate(texts, 1):
    print(f"[{i}/{len(texts)}] Generating: '{text[:50]}...'")
    text_start = time.time()
    
    # ❌ This prepares conditionals every time (inefficient!)
    wav = model.generate(
        text,
        language_id=language_id,
        audio_prompt_path=AUDIO_PROMPT_PATH,  # Triggers prepare_conditionals()
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )
    
    text_elapsed = time.time() - text_start
    print(f"    Time: {text_elapsed:.2f}s\n")

without_cache_time = time.time() - start_time
print(f"Total time WITHOUT caching: {without_cache_time:.2f}s\n")


print("=" * 80)
print("METHOD 2: WITH CACHING (Efficient)")
print("=" * 80)
print("Preparing conditionals ONCE, then reusing for all texts...\n")

# ✅ Prepare conditionals once
print(f"Preparing conditionals from: {AUDIO_PROMPT_PATH}")
prep_start = time.time()
model.prepare_conditionals(AUDIO_PROMPT_PATH, exaggeration=exaggeration)
prep_time = time.time() - prep_start
print(f"Conditionals prepared in {prep_time:.2f}s\n")

# Optional: Save conditionals to disk for later use
cache_file = Path("cached_conditionals.pt")
model.conds.save(cache_file)
print(f"Conditionals saved to: {cache_file}\n")

start_time = time.time()
for i, text in enumerate(texts, 1):
    print(f"[{i}/{len(texts)}] Generating: '{text[:50]}...'")
    text_start = time.time()
    
    # ✅ No audio_prompt_path = reuses cached conditionals!
    wav = model.generate(
        text,
        language_id=language_id,
        # audio_prompt_path=None,  # Not specified - uses cached conditionals
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )
    
    text_elapsed = time.time() - text_start
    print(f"    Time: {text_elapsed:.2f}s\n")
    
    # Save the last one as example
    if i == len(texts):
        ta.save("example_cached_output.wav", wav, model.sr)
        print(f"    Saved to: example_cached_output.wav\n")

with_cache_time = time.time() - start_time
print(f"Total generation time WITH caching: {with_cache_time:.2f}s")
print(f"(Excluding one-time preparation: {prep_time:.2f}s)\n")


print("=" * 80)
print("METHOD 3: LOADING CACHED CONDITIONALS FROM DISK")
print("=" * 80)
print("Demonstrating how to load pre-saved conditionals...\n")

# ✅ Load conditionals from disk (even faster - skip preparation!)
if cache_file.exists():
    from chatterbox.mtl_tts import Conditionals
    
    print(f"Loading conditionals from: {cache_file}")
    load_start = time.time()
    model.conds = Conditionals.load(cache_file, map_location=device)
    model.conds = model.conds.to(device)
    load_time = time.time() - load_start
    print(f"Conditionals loaded in {load_time:.2f}s\n")
    
    # Generate with loaded conditionals
    test_text = "This uses conditionals loaded from disk!"
    print(f"Generating: '{test_text}'")
    gen_start = time.time()
    wav = model.generate(
        test_text,
        language_id=language_id,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )
    gen_time = time.time() - gen_start
    print(f"Generated in {gen_time:.2f}s\n")


print("=" * 80)
print("PERFORMANCE SUMMARY")
print("=" * 80)
print(f"Method 1 (No caching):        {without_cache_time:.2f}s total")
print(f"Method 2 (With caching):      {prep_time:.2f}s prep + {with_cache_time:.2f}s generation = {prep_time + with_cache_time:.2f}s total")
print(f"Method 3 (Load from disk):    {load_time:.2f}s load + {gen_time:.2f}s generation\n")

speedup = without_cache_time / (with_cache_time + prep_time) if with_cache_time + prep_time > 0 else 0
print(f"Speedup: {speedup:.2f}x faster with caching!")
print(f"\n✅ For {len(texts)} texts, caching saves ~{without_cache_time - with_cache_time:.2f}s\n")


print("=" * 80)
print("CACHING STRATEGY GUIDE")
print("=" * 80)
print("""
When to cache conditionals:
✅ Same reference audio (voice) for multiple texts
✅ Same exaggeration parameter
✅ Batch processing multiple texts
✅ Web applications with pre-defined voices

When NOT to cache:
❌ Different reference audio files
❌ Different exaggeration values
❌ Single text generation

Implementation tips:
1. For production: Create a dictionary cache keyed by (audio_path, exaggeration)
2. Save commonly-used conditionals to disk
3. Load on startup for frequently-used voices
4. Memory: Each cached conditional is ~few MB (small!)
""")


print("\nExample cache dictionary implementation:")
print("-" * 80)
print("""
# Advanced caching with multiple voices:
conditionals_cache = {}

def get_or_create_conditionals(model, audio_path, exaggeration):
    cache_key = (audio_path, exaggeration)
    
    if cache_key not in conditionals_cache:
        print(f"Cache miss - preparing conditionals for {audio_path}")
        model.prepare_conditionals(audio_path, exaggeration)
        conditionals_cache[cache_key] = model.conds
    else:
        print(f"Cache hit - reusing conditionals for {audio_path}")
        model.conds = conditionals_cache[cache_key]
    
    return model.conds

# Usage:
get_or_create_conditionals(model, "voice1.wav", 0.2)
wav1 = model.generate("Text 1", language_id="en")
wav2 = model.generate("Text 2", language_id="en")  # Reuses cache

get_or_create_conditionals(model, "voice2.wav", 0.5)
wav3 = model.generate("Text 3", language_id="en")  # Different voice
""")

print("\n" + "=" * 80)
print("Done! Check the generated files:")
print("  - example_cached_output.wav")
print("  - cached_conditionals.pt")
print("=" * 80)
