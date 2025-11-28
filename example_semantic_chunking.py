"""
Example of using SmolLM3 semantic chunking for long-form TTS generation.

This demonstrates the difference between simple regex-based chunking and
AI-powered semantic chunking for better audio quality in long texts.
"""

import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from dotenv import load_dotenv
import os

load_dotenv()

# Access the variables
hf_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# Load the multilingual model
model = ChatterboxMultilingualTTS.from_pretrained(device=device)

# Example long text with different semantic sections
LONG_TEXT = """
The old lighthouse stood at the edge of the cliff, its weathered stones bearing witness to countless storms.
For over a century, it had guided ships safely to harbor, its beam cutting through fog and darkness alike.

But times had changed. Modern navigation systems had made lighthouses obsolete, and one by one, they were
being decommissioned. The lighthouse keeper, an elderly man named Thomas, knew his days there were numbered.

Thomas walked slowly up the spiral staircase, his footsteps echoing in the empty tower. He had climbed
these stairs every day for forty years, tending to the light that had been his life's work. Each step
brought back memories - of wild storms, of ships saved, of the solitary beauty of life by the sea.

At the top, he paused to look out over the ocean. The sun was setting, painting the sky in shades of
orange and pink. It would be his last sunset from this vantage point. Tomorrow, the lighthouse would
be officially closed, and Thomas would return to a life on solid ground.

He lit the lamp one final time, watching as its beam swept across the darkening waters. It was a small
act of defiance, a way of saying goodbye to the sea that had been his companion for so long.
"""

# Reference audio for voice cloning
AUDIO_PROMPT_PATH = "earl-nightingale.wav"  # Replace with your audio file


def progress_callback(stage, **kwargs):
    """Callback to monitor generation progress."""
    if stage == "text_split":
        print(f"\n✓ Split into {kwargs['total_chunks']} chunks")
    elif stage == "chunk_start":
        print(f"  → Chunk {kwargs['chunk_number']}/{kwargs['total_chunks']}: {kwargs['word_count']} words")
    elif stage == "crossfading":
        print(f"\n✓ Crossfading {kwargs['total_chunks']} audio chunks...")
    elif stage == "complete":
        print(f"✓ Complete! Final audio shape: {kwargs['final_audio_shape']}")


print("\n" + "="*80)
print("COMPARISON: Simple vs Semantic Chunking")
print("="*80)

# Example 3: Standalone semantic chunker usage
print("\n[3] Standalone Semantic Chunker")
print("-" * 80)
print("You can also use the semantic chunker independently:")
print()

try:
    from chatterbox.semantic_chunker import create_semantic_chunks

    # Create chunks without generating audio
    chunks = create_semantic_chunks(
        LONG_TEXT,
        language_id="en",
        target_words=50,
        device=device
    )

    print(f"Generated {len(chunks)} semantic chunks:")
    for i, chunk in enumerate(chunks, 1):
        word_count = len(chunk.split())
        preview = chunk[:80] + "..." if len(chunk) > 80 else chunk
        print(f"  {i}. ({word_count} words) {preview}")

except ImportError:
    print("  Semantic chunker not available (requires transformers>=4.53.0)")

# Example 2: Semantic chunking with SmolLM3
print("\n[2] Semantic Chunking (SmolLM3-powered)")
print("-" * 80)
try:
    wav_semantic = model.generate_long(
        LONG_TEXT,
        language_id="en",
        audio_prompt_path=AUDIO_PROMPT_PATH,
        chunk_size_words=50,
        overlap_duration=1.0,
        use_semantic_chunking=True,  # Enable AI-powered chunking
        progress_callback=progress_callback,
        token=hf_token,
    )
    ta.save("output_semantic_chunking.wav", wav_semantic, model.sr)
    print("✓ Saved to: output_semantic_chunking.wav")
except RuntimeError as e:
    print(f"⚠ Semantic chunking not available: {e}")
    print("  To enable: pip install 'transformers>=4.53.0'")


# Example 1: Simple regex-based chunking (default)
print("\n[1] Simple Chunking (regex-based)")
print("-" * 80)
wav_simple = model.generate_long(
    LONG_TEXT,
    language_id="en",
    audio_prompt_path=AUDIO_PROMPT_PATH,
    chunk_size_words=50,
    overlap_duration=1.0,
    use_semantic_chunking=False,  # Default behavior
    progress_callback=progress_callback,
    token=hf_token,
)
ta.save("output_simple_chunking.wav", wav_simple, model.sr)
print("✓ Saved to: output_simple_chunking.wav")



print("\n" + "="*80)
print("Comparison Summary")
print("="*80)
print("""
Simple Chunking:
  ✓ Fast (no AI model loading)
  ✓ Predictable chunk sizes
  ✗ May split at awkward points (mid-topic, mid-scene)
  ✗ Doesn't understand semantic context

Semantic Chunking (SmolLM3):
  ✓ Understands topic boundaries and transitions
  ✓ Respects dialogue turns and scene changes
  ✓ More natural-sounding speech flow
  ✗ Slower (AI model inference required)
  ✗ Requires more memory (3B parameter model)

Recommendation:
  - Use simple chunking for quick tests or when speed matters
  - Use semantic chunking for production/high-quality audiobooks
""")

print("✓ All examples completed!")
