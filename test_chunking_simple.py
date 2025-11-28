"""
Simple test script for semantic chunking functionality.
Tests the chunker without loading the full TTS pipeline.
"""

import torch
import os
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

# Simple test text
TEST_TEXT = """
The old lighthouse stood at the edge of the cliff, its weathered stones bearing witness to countless storms.
For over a century, it had guided ships safely to harbor, its beam cutting through fog and darkness alike.

But times had changed. Modern navigation systems had made lighthouses obsolete, and one by one, they were
being decommissioned. The lighthouse keeper, an elderly man named Thomas, knew his days there were numbered.

Thomas walked slowly up the spiral staircase, his footsteps echoing in the empty tower. He had climbed
these stairs every day for forty years, tending to the light that had been his life's work.
"""

print("="*80)
print("SEMANTIC CHUNKING TEST")
print("="*80)

# Detect device (force CPU to avoid MPS issues)
device = "cpu"
print(f"\nUsing device: {device}")
print(f"Test text length: {len(TEST_TEXT)} chars, {len(TEST_TEXT.split())} words\n")

# Test the semantic chunker
try:
    from chatterbox.semantic_chunker import create_semantic_chunks

    print("-" * 80)
    print("Testing semantic chunking with SmolLM3...")
    print("-" * 80)

    chunks = create_semantic_chunks(
        TEST_TEXT,
        language_id="en",
        target_words=50,  # Smaller chunks for faster testing
        device=device,
        token=hf_token,
    )

    print("\n" + "="*80)
    print(f"RESULTS: Generated {len(chunks)} chunks")
    print("="*80)

    for i, chunk in enumerate(chunks, 1):
        word_count = len(chunk.split())
        print(f"\nChunk {i} ({word_count} words):")
        print("-" * 40)
        print(chunk)
        print("-" * 40)

    # Verify all text is preserved
    reconstructed = " ".join(chunks)
    original_words = TEST_TEXT.split()
    reconstructed_words = reconstructed.split()

    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    print(f"Original words: {len(original_words)}")
    print(f"Reconstructed words: {len(reconstructed_words)}")

    if len(chunks) > 1:
        print("✓ Text was successfully split into multiple chunks")
    else:
        print("⚠ Text was not split (returned as single chunk)")

except ImportError as e:
    print(f"✗ Error: {e}")
    print("Make sure transformers>=4.53.0 is installed")
except Exception as e:
    print(f"✗ Error during chunking: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("Test complete!")
print("="*80)
