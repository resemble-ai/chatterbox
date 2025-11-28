# Advanced Usage

## Voice Caching

When generating multiple texts with the same voice, cache the voice conditionals to avoid redundant processing.

### Quick Start

```python
from conditional_cache import ConditionalCache

cache = ConditionalCache(cache_dir="./tts_cache", auto_save=True)

# Cache is used automatically
for text in texts:
    cache.get_or_prepare(model, "voice.wav", exaggeration=0.5)
    wav = model.generate(text, language_id="en", exaggeration=0.5)
```

**Important**: Exaggeration must match between `get_or_prepare()` and `generate()`. Don't pass `audio_prompt_path` to `generate()` when using cached conditionals.

### Performance

- Without caching: ~40-60s for 4 texts
- With caching: ~15-20s (2-3x faster)
- Cached files: ~5-20 MB per voice

See `example_simple_caching.py` for complete examples.

---

## Semantic Chunking for Long Text

For long-form TTS (audiobooks, articles), use AI-powered semantic chunking to split text at natural boundaries.

### Requirements

```bash
pip install 'transformers>=4.53.0'
```

### Usage

```python
wav = model.generate_long(
    long_text,
    language_id="en",
    audio_prompt_path="voice.wav",
    use_semantic_chunking=True,  # Uses SmolLM3 for intelligent splitting
    chunk_size_words=50,
)
```

### When to Use

| Feature  | Simple Chunking | Semantic Chunking |
| -------- | --------------- | ----------------- |
| Speed    | Fast            | Slower            |
| Quality  | Good            | Excellent         |
| Use case | Prototyping     | Production        |

### Best For

- Audiobook narration
- Multi-paragraph articles
- Dialogue-heavy content

See `example_semantic_chunking.py` for complete examples.

---

## Audiobook Generation

Combine caching + chunking for multi-voice audiobook production:

```python
cache = ConditionalCache(cache_dir="./cache", auto_save=True)

voices = {
    "narrator": ("narrator.wav", 0.3),
    "character": ("character.wav", 0.7),
}

for speaker, line in script:
    voice_file, exag = voices[speaker]
    cache.get_or_prepare(model, voice_file, exaggeration=exag)
    wav = model.generate(line, language_id="en", exaggeration=exag)
```

See `example_audiobook.py` for a complete implementation.
