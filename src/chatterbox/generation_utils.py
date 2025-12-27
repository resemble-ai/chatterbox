# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
Shared utilities for TTS generation across all backends.

This module provides common functionality for:
- Sentence splitting (spacy-based with caching)
- Crossfade/concatenation of audio chunks
- Adaptive chunking strategies
- Generation observability (progress logging)
"""

from typing import List
import re
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ============================================================================
# Spacy Sentence Splitting
# ============================================================================

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Cache for loaded spacy models
_spacy_models = {}


def _get_spacy_model(lang: str = "en"):
    """
    Get or load a spacy model for the given language.
    Uses a cache to avoid reloading models.

    Args:
        lang: Language code (e.g., "en", "de", "fr")

    Returns:
        Loaded spacy model with sentencizer
    """
    if not SPACY_AVAILABLE:
        return None

    if lang not in _spacy_models:
        try:
            # Try to load a language-specific model first
            model_map = {
                "en": "en_core_web_sm",
                "de": "de_core_news_sm",
                "fr": "fr_core_news_sm",
                "es": "es_core_news_sm",
                "it": "it_core_news_sm",
                "pt": "pt_core_news_sm",
                "nl": "nl_core_news_sm",
                "pl": "pl_core_news_sm",
                "ru": "ru_core_news_sm",
                "ja": "ja_core_news_sm",
                "zh": "zh_core_web_sm",
                "el": "el_core_news_sm",
                "da": "da_core_news_sm",
                "fi": "fi_core_news_sm",
                "ko": "ko_core_news_sm",
                "sv": "sv_core_news_sm",
                "no": "nb_core_news_sm",
            }

            model_name = model_map.get(lang)
            if model_name:
                try:
                    nlp = spacy.load(model_name)
                    _spacy_models[lang] = nlp
                    logger.info(f"Loaded spacy model: {model_name}")
                except OSError:
                    # Model not installed, use blank model with sentencizer
                    logger.warning(
                        f"Spacy model {model_name} not found, using blank {lang} model with sentencizer"
                    )
                    nlp = spacy.blank(lang)
                    nlp.add_pipe("sentencizer")
                    _spacy_models[lang] = nlp
            else:
                # Language not in map, use blank model
                logger.info(
                    f"No specific spacy model for {lang}, using blank model with sentencizer"
                )
                nlp = spacy.blank(lang)
                nlp.add_pipe("sentencizer")
                _spacy_models[lang] = nlp

        except Exception as e:
            logger.warning(
                f"Failed to load spacy model for {lang}: {e}, falling back to English"
            )
            if "en" not in _spacy_models:
                nlp = spacy.blank("en")
                nlp.add_pipe("sentencizer")
                _spacy_models["en"] = nlp
            _spacy_models[lang] = _spacy_models["en"]

    return _spacy_models[lang]


def split_into_sentences(text: str, lang: str = "en") -> List[str]:
    """
    Split text into sentences using spacy (preferred) or regex fallback.

    Args:
        text: Input text to split
        lang: Language code for the spacy model

    Returns:
        List of sentence strings
    """
    if SPACY_AVAILABLE:
        nlp = _get_spacy_model(lang)
        if nlp is not None:
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            return sentences if sentences else [text]

    # Fallback to regex if spacy is not available
    logger.warning("Spacy not available, falling back to regex sentence splitting")
    sentence_pattern = r"(?<=[.!?])\s+"
    return [s.strip() for s in re.split(sentence_pattern, text) if s.strip()] or [text]


# ============================================================================
# Adaptive Chunking
# ============================================================================

# Thresholds for adaptive chunking strategy
ADAPTIVE_THRESHOLD_WORDS = 50  # Below this, use per-sentence; above, use grouped
TARGET_WORDS_PER_CHUNK = 35  # Target word count when grouping sentences


def get_adaptive_chunks(
    text: str,
    lang: str = "en",
    threshold_words: int = ADAPTIVE_THRESHOLD_WORDS,
    target_words_per_chunk: int = TARGET_WORDS_PER_CHUNK,
) -> tuple[List[str], str]:
    """
    Get text chunks using adaptive strategy based on total word count.

    For short texts (< threshold_words): process each sentence individually
    For long texts (>= threshold_words): group sentences to reduce overhead

    Args:
        text: Input text to chunk
        lang: Language code for sentence splitting
        threshold_words: Word count threshold for switching strategies
        target_words_per_chunk: Target words per chunk when grouping

    Returns:
        Tuple of (chunks list, strategy description string)
    """
    # Split into individual sentences
    sentences = split_into_sentences(text, lang=lang)
    if not sentences:
        sentences = [text]

    total_words = len(text.split())

    if total_words < threshold_words:
        # Short text: process each sentence individually
        return sentences, "per-sentence"
    else:
        # Long text: group sentences to reduce per-chunk overhead
        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            # If adding this sentence would exceed target, flush current chunk
            if (
                current_word_count > 0
                and current_word_count + sentence_words > target_words_per_chunk
            ):
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_word_count = sentence_words
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_words

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return (
            chunks if chunks else [text]
        ), f"grouped (~{target_words_per_chunk} words/chunk)"


def split_text_intelligently(
    text: str,
    target_words_per_chunk: int = 50,
    lang: str = "en",
) -> List[str]:
    """
    Split text at sentence boundaries, grouping to target word count.

    Args:
        text: Input text to split
        target_words_per_chunk: Target number of words per chunk
        lang: Language code for sentence splitting

    Returns:
        List of text chunks
    """
    sentences = split_into_sentences(text, lang=lang)

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        if not sentence.strip():
            continue
        word_count = len(sentence.split())

        if current_word_count + word_count > target_words_per_chunk and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks if chunks else [text]


# ============================================================================
# Audio Crossfade
# ============================================================================


def crossfade_chunks(
    chunks: List[torch.Tensor],
    sample_rate: int,
    overlap_duration: float = 0.1,
) -> torch.Tensor:
    """
    Concatenate audio chunks with crossfading using optimized vectorized operations.

    This implementation is optimized for many chunks by:
    1. Pre-allocating the output array
    2. Using vectorized numpy operations
    3. Pre-computing fade curves once

    Args:
        chunks: List of audio torch tensors
        sample_rate: Audio sample rate (e.g., 24000)
        overlap_duration: Duration of crossfade in seconds

    Returns:
        Single concatenated audio tensor (1D)
    """
    if len(chunks) == 0:
        return torch.tensor([], dtype=torch.float32)
    if len(chunks) == 1:
        chunk = chunks[0]
        if chunk.dim() == 2:
            chunk = chunk.squeeze(0)
        return chunk

    overlap_samples = int(overlap_duration * sample_rate)

    # Convert all chunks to numpy and flatten
    processed = []
    for chunk in chunks:
        if isinstance(chunk, torch.Tensor):
            chunk = chunk.cpu().numpy()
        if chunk.ndim == 2:
            chunk = chunk.squeeze(0)
        processed.append(chunk)

    # For small number of chunks, use simple approach
    if len(processed) <= 3:
        result = processed[0]
        for chunk in processed[1:]:
            if len(result) > overlap_samples and len(chunk) > overlap_samples:
                fade_out = np.linspace(1, 0, overlap_samples)
                fade_in = np.linspace(0, 1, overlap_samples)
                result_end = result[-overlap_samples:] * fade_out
                chunk_start = chunk[:overlap_samples] * fade_in
                result = np.concatenate(
                    [
                        result[:-overlap_samples],
                        result_end + chunk_start,
                        chunk[overlap_samples:],
                    ]
                )
            else:
                result = np.concatenate([result, chunk])
        return torch.from_numpy(result)

    # Optimized approach for many chunks: pre-allocate and batch process
    total_length = sum(len(c) for c in processed) - overlap_samples * (
        len(processed) - 1
    )
    result = np.zeros(total_length, dtype=np.float32)

    # Pre-compute fade curves once
    fade_out = np.linspace(1, 0, overlap_samples, dtype=np.float32)
    fade_in = np.linspace(0, 1, overlap_samples, dtype=np.float32)

    # Place first chunk
    current_pos = 0
    first_chunk = processed[0]
    result[: len(first_chunk)] = first_chunk
    current_pos = len(first_chunk) - overlap_samples

    # Process remaining chunks with crossfade
    for chunk in processed[1:]:
        if overlap_samples > 0 and len(chunk) > overlap_samples:
            # Apply crossfade in-place
            result[current_pos : current_pos + overlap_samples] *= fade_out
            result[current_pos : current_pos + overlap_samples] += (
                chunk[:overlap_samples] * fade_in
            )
            rest_start = current_pos + overlap_samples
            rest_length = len(chunk) - overlap_samples
            result[rest_start : rest_start + rest_length] = chunk[overlap_samples:]
            current_pos = rest_start + rest_length - overlap_samples
        else:
            result[current_pos : current_pos + len(chunk)] = chunk
            current_pos += len(chunk)

    return torch.from_numpy(result)


# ============================================================================
# max_new_tokens Estimation
# ============================================================================


def estimate_max_tokens(text: str, model_max: int = 4096) -> int:
    """
    Estimate reasonable max_new_tokens based on text length.

    Uses word-based estimation:
    - Average speaking rate: ~150 words/minute = 2.5 words/second
    - Speech tokens: 25 tokens/second
    - Therefore: ~10 tokens per word on average
    - Safety buffer: 1.5x to handle slow speech and pauses

    Args:
        text: Input text
        model_max: Maximum tokens the model supports

    Returns:
        Estimated max_new_tokens value
    """
    word_count = len(text.split())
    # ~10 tokens per word, with 1.5x safety buffer
    estimated_tokens = word_count * 10 * 1.5
    return min(
        max(int(estimated_tokens), 100),  # At least 100 tokens minimum
        model_max,  # Never exceed model max
    )


# ============================================================================
# Observability / Progress Logging
# ============================================================================


def print_generation_plan(
    total_words: int,
    chunks: List[str],
    strategy: str = "per-sentence",
    is_long_form: bool = False,
    prefix: str = "",
) -> None:
    """
    Print the generation plan overview.

    Args:
        total_words: Total word count
        chunks: List of text chunks to generate
        strategy: Chunking strategy description
        is_long_form: Whether this is long-form generation
        prefix: Optional prefix for log lines (e.g., "[PureMLX] ")
    """
    num_chunks = len(chunks)
    title = "LONG-FORM GENERATION PLAN" if is_long_form else "GENERATION PLAN"
    if num_chunks == 1:
        title += " (single chunk)"

    print(f"\n{'='*60}")
    print(f"📝 {prefix}{title}")
    print(f"{'='*60}")
    print(f"  Total words: {total_words}")
    if num_chunks > 1:
        print(f"  Strategy: {strategy}")
        print(f"  Chunks to generate: {num_chunks}")
        print(f"{'─'*60}")
        print("  CHUNKS OVERVIEW:")
        for i, chunk in enumerate(chunks):
            chunk_words = len(chunk.split())
            preview = chunk[:60].replace("\n", " ") + ("..." if len(chunk) > 60 else "")
            print(
                f'    [{i+1}/{num_chunks}] ⏳ pending | {chunk_words:>3} words | "{preview}"'
            )
    else:
        preview = chunks[0][:50].replace("\n", " ") + (
            "..." if len(chunks[0]) > 50 else ""
        )
        print(f'  Text: "{preview}"')
    print(f"{'='*60}\n")


def print_chunk_generating(
    chunk_idx: int,
    total_chunks: int,
    chunk_text: str,
) -> None:
    """Print status when starting to generate a chunk."""
    chunk_words = len(chunk_text.split())
    preview = chunk_text[:40].replace("\n", " ") + (
        "..." if len(chunk_text) > 40 else ""
    )
    print(
        f'  🔄 [{chunk_idx+1}/{total_chunks}] GENERATING: "{preview}" ({chunk_words} words)'
    )


def print_chunk_completed(
    chunk_idx: int,
    total_chunks: int,
    gen_time: float,
    audio_duration: float,
) -> None:
    """Print status when a chunk is completed."""
    realtime_factor = audio_duration / gen_time if gen_time > 0 else 0
    print(
        f"  ✅ [{chunk_idx+1}/{total_chunks}] COMPLETED: {gen_time:.1f}s generation → {audio_duration:.1f}s audio ({realtime_factor:.2f}x realtime)"
    )


def print_generation_complete(
    total_time: float,
    total_audio_duration: float,
    num_chunks: int = 1,
    prefix: str = "",
) -> None:
    """Print final generation summary."""
    realtime_factor = total_audio_duration / total_time if total_time > 0 else 0
    print(f"\n{'='*60}")
    print(f"✅ {prefix}GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total generation time: {total_time:.1f}s")
    print(f"  Total audio duration:  {total_audio_duration:.1f}s")
    print(f"  Overall realtime factor: {realtime_factor:.2f}x")
    if num_chunks > 1:
        print(f"  Chunks generated: {num_chunks}")
    print(f"{'='*60}\n")


def print_crossfading(num_chunks: int) -> None:
    """Print crossfading status."""
    print(f"\n  🔗 Crossfading {num_chunks} chunks...")
