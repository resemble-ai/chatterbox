# Conditional Caching for Chatterbox TTS

This guide explains how to efficiently cache audio conditionals when generating multiple texts with the same reference voice.

## Why Cache Conditionals?

When you generate audio with Chatterbox TTS, the system processes your reference audio file to create "conditionals" - embeddings and tokens that represent the voice characteristics. This processing is:

- **Expensive**: Takes significant time and memory
- **Redundant**: The same reference audio produces identical conditionals
- **Text-independent**: Conditionals depend ONLY on the audio file and exaggeration, NOT on the text being synthesized

**Key insight**: If you're generating multiple different texts with the same voice, you're wasting time re-processing the same audio!

## Performance Impact

For generating 4 texts with the same voice:

- **Without caching**: ~40-60 seconds (prepares conditionals 4 times)
- **With caching**: ~15-20 seconds (prepares conditionals once, reuses 3 times)
- **Speedup**: 2-3x faster!

## Quick Start

### Option 1: Manual Caching (Simple)

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")

# Prepare conditionals ONCE
model.prepare_conditionals("voice.wav", exaggeration=0.5)

# Generate multiple texts (fast - reuses conditionals!)
wav1 = model.generate("Text one", language_id="en")
wav2 = model.generate("Text two", language_id="en")
wav3 = model.generate("Text three", language_id="en")
```

### Option 2: Using ConditionalCache (Recommended)

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from conditional_cache import ConditionalCache

model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
cache = ConditionalCache(cache_dir="./tts_cache", auto_save=True)

texts = ["Text one", "Text two", "Text three"]

for text in texts:
    # Automatically caches and reuses conditionals
    cache.get_or_prepare(model, "voice.wav", exaggeration=0.5)
    wav = model.generate(text, language_id="en")
```

## Files Included

### Example Scripts

1. **`example_simple_caching.py`** - Easiest way to get started

   - Uses the ConditionalCache utility
   - Shows how to generate multiple texts efficiently
   - Demonstrates multi-voice caching

2. **`example_with_caching.py`** - Comprehensive demonstration

   - Compares performance with/without caching
   - Shows disk persistence
   - Includes timing benchmarks

3. **`conditional_cache.py`** - Production-ready utility
   - Memory and disk caching
   - Automatic cache management
   - Multi-voice support

## How It Works

### What are Conditionals?

Conditionals are derived from your **reference audio file** and contain:

1. **Voice embeddings** - Neural representation of the speaker's voice
2. **Speech tokens** - Tokenized version of the reference audio
3. **Exaggeration parameter** - Controls expressiveness

These are created by `prepare_conditionals(audio_path, exaggeration)`.

### Cache Key

Conditionals are uniquely identified by:

- Reference audio file path
- Exaggeration value

**Same audio + same exaggeration = Same conditionals = Can be cached!**

### What Gets Cached?

The `Conditionals` object containing:

- T3 model conditionals (speaker_emb, cond_prompt_speech_tokens, emotion_adv)
- S3Gen model conditionals (prompt_token, prompt_feat, embeddings)

File size: ~5-20 MB per cached voice

## Usage Patterns

### Pattern 1: Batch Processing Same Voice

```python
cache = ConditionalCache(cache_dir="./cache")

articles = load_articles()  # Many texts
voice = "narrator.wav"

for article in articles:
    cache.get_or_prepare(model, voice, exaggeration=0.5)
    wav = model.generate(article, language_id="en")
    save_audio(wav)
```

### Pattern 2: Multiple Voices

```python
cache = ConditionalCache(cache_dir="./cache")

voices = {
    "narrator": ("narrator.wav", 0.3),
    "character1": ("char1.wav", 0.7),
    "character2": ("char2.wav", 0.5),
}

for speaker, line in dialog:
    voice_file, exag = voices[speaker]
    cache.get_or_prepare(model, voice_file, exaggeration=exag)
    wav = model.generate(line, language_id="en")
```

### Pattern 3: Pre-warming Cache

```python
# Load all commonly-used voices into cache on startup
cache = ConditionalCache(cache_dir="./cache")

common_voices = [
    ("voice1.wav", 0.5),
    ("voice2.wav", 0.3),
    ("voice3.wav", 0.7),
]

print("Pre-warming cache...")
for voice_file, exag in common_voices:
    cache.get_or_prepare(model, voice_file, exaggeration=exag)

print("Cache ready! All subsequent generations will be fast.")
```

## ConditionalCache API

### Constructor

```python
cache = ConditionalCache(
    cache_dir="./cache",  # Where to store cached files (None = memory only)
    auto_save=True        # Automatically save new conditionals to disk
)
```

### Main Methods

```python
# Get or create conditionals (automatic caching)
cache.get_or_prepare(model, "voice.wav", exaggeration=0.5)

# Force refresh (ignore cache)
cache.get_or_prepare(model, "voice.wav", exaggeration=0.5, force_refresh=True)

# Save specific cached conditionals
cache.save("voice.wav", 0.5, filepath="my_voice.pt")

# Load conditionals from file
cache.load("my_voice.pt", "voice.wav", 0.5, device="cpu")

# Cache management
cache.clear_memory()  # Clear RAM cache
cache.clear_disk()    # Delete all cache files
cache.clear_all()     # Clear both

# Statistics
stats = cache.get_cache_stats()
print(cache)  # Shows cache status
```

## Best Practices

### ✅ DO:

- Cache when generating multiple texts with the same voice
- Use disk caching for commonly-used voices
- Pre-warm cache for production applications
- Clear cache when switching to different audio files

### ❌ DON'T:

- Cache when generating single texts
- Mix exaggeration values (each creates a different cache entry)
- Forget to match exaggeration between `get_or_prepare()` and `generate()`

## Common Mistakes

### Mistake 1: Not matching exaggeration values

```python
# ❌ WRONG - exaggeration mismatch
cache.get_or_prepare(model, "voice.wav", exaggeration=0.5)
wav = model.generate(text, language_id="en", exaggeration=0.7)  # Different!
```

```python
# ✅ CORRECT - matching exaggeration
exag = 0.5
cache.get_or_prepare(model, "voice.wav", exaggeration=exag)
wav = model.generate(text, language_id="en", exaggeration=exag)
```

### Mistake 2: Still passing audio_prompt_path

```python
# ❌ WRONG - this ignores cache and re-prepares!
cache.get_or_prepare(model, "voice.wav", exaggeration=0.5)
wav = model.generate(text, language_id="en", audio_prompt_path="voice.wav")
```

```python
# ✅ CORRECT - omit audio_prompt_path to use cached conditionals
cache.get_or_prepare(model, "voice.wav", exaggeration=0.5)
wav = model.generate(text, language_id="en")  # Uses cached conditionals
```

## Performance Tips

1. **Disk cache persistence**: Conditionals saved to disk survive program restarts
2. **Memory usage**: Each cached voice uses ~5-20 MB RAM
3. **Startup optimization**: Load frequently-used voices on startup
4. **Multi-processing**: Each process needs its own cache instance

## Running the Examples

```bash
# Simple caching example (recommended starting point)
python example_simple_caching.py

# Comprehensive comparison with benchmarks
python example_with_caching.py
```

## Cache Directory Structure

```
./tts_cache/
├── cond_julia-whelan_a1b2c3d4_exag0_5000.pt
├── cond_earl-nightingale_e5f6g7h8_exag0_3000.pt
└── cond_narrator_i9j0k1l2_exag0_7000.pt
```

Each file contains one cached `Conditionals` object for a specific (audio_file, exaggeration) combination.

## FAQ

**Q: Can I share cache files between machines?**
A: Yes! The `.pt` files are portable. Just ensure the audio file paths match or update the cache key.

**Q: How much disk space do I need?**
A: Each cached voice is ~5-20 MB. For 10 voices, expect ~100-200 MB total.

**Q: Does caching affect audio quality?**
A: No! Cached conditionals produce identical audio to freshly-prepared ones.

**Q: Can I cache different exaggeration values for the same voice?**
A: Yes! Each (voice, exaggeration) pair gets its own cache entry.

**Q: What if my audio file changes?**
A: Use `force_refresh=True` or clear the cache to regenerate conditionals.

## Troubleshooting

### Cache not being used

Check that:

1. Exaggeration values match between `get_or_prepare()` and `generate()`
2. You're NOT passing `audio_prompt_path` to `generate()`
3. Audio file paths are consistent (use absolute paths)

### Memory issues

- Limit number of cached voices
- Use `cache.clear_memory()` periodically
- Disable auto_save if disk space is limited

### Cache files corrupted

```python
cache.clear_disk()  # Delete all cache files
cache.clear_memory()  # Clear memory
# Regenerate as needed
```

## License

Same as Chatterbox TTS main project.
