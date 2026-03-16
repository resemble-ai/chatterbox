# src/chatterbox/word_timestamps.py
#
# Word-level timestamp extraction from Chatterbox TTS.
#
# Derives per-word timing from the proportional relationship between
# text tokens and speech tokens � the same principle timed_tts.py uses
# to split segments, applied one level deeper at word granularity.
#
# Each speech token = exactly 40ms (TOKEN_TO_WAV_RATIO / S3GEN_SR = 960 / 24000).
#
# NOTE on [SPACE] tokens: the full-sentence tokenizer inserts [SPACE] tokens
# between words, but per-word tokenization does not produce them. This means
# sum(per_word_counts) < full_sentence_count. The ~40ms per [SPACE] gets
# distributed evenly across words, which is acceptable at this resolution.
#
# NOTE on punc_norm: words in the output reflect the normalized text
# (e.g. "..." becomes ", ") because we must tokenize the same string
# the TTS pipeline sees. Mapping back to original words is fragile
# when punc_norm changes word count, so we accept normalized words.

import json
from dataclasses import dataclass, asdict
from typing import List, Optional

from .models.s3gen.const import S3GEN_SR, TOKEN_TO_WAV_RATIO
from .models.tokenizers import SPACE
from .tts import punc_norm

SECS_PER_TOKEN = TOKEN_TO_WAV_RATIO / S3GEN_SR  # 0.04s = 40ms


@dataclass
class WordTimestamp:
    """Timing info for a single word."""
    word: str
    start: float       # seconds
    end: float         # seconds
    text_tokens: int   # how many text tokens this word consumed
    speech_tokens: int # estimated speech tokens for this word


def _count_word_tokens(tokenizer, word: str) -> int:
    """
    Count text tokens for a single word using the raw HuggingFace
    Tokenizer.encode() with [SPACE] replacement.

    Does NOT include inter-word [SPACE] tokens � see module docstring.
    """
    processed = word.replace(' ', SPACE)
    return max(len(tokenizer.tokenizer.encode(processed).ids), 1)


def _split_words(text: str) -> List[str]:
    """Split on whitespace. Punctuation stays attached to its word."""
    return [w for w in text.split() if w]


def extract_word_timestamps(
    tokenizer,
    text: str,
    n_speech_tokens: int,
    time_offset: float = 0.0,
    apply_punc_norm: bool = True,
) -> List[WordTimestamp]:
    """
    Extract word-level timestamps from TTS generation results.

    Parameters
    ----------
    tokenizer : EnTokenizer or MTLTokenizer
        The text tokenizer used by ChatterboxTTS.
    text : str
        The original text that was synthesized.
    n_speech_tokens : int
        Number of speech tokens T3 generated for this text.
    time_offset : float
        Base time offset (for segments placed at specific positions).
    apply_punc_norm : bool
        Whether to apply punc_norm before tokenizing (should match
        what the TTS pipeline does).

    Returns
    -------
    List[WordTimestamp]
        One entry per word with start/end times in seconds.
        Words reflect the punc_norm'd text, not the original input.
    """
    if n_speech_tokens <= 0:
        return []

    normed = punc_norm(text) if apply_punc_norm else text
    words = _split_words(normed)

    if not words:
        return []

    # Count text tokens per word � mirrors timed_tts.py's proportional
    # split but at word granularity instead of segment granularity.
    token_counts = [_count_word_tokens(tokenizer, w) for w in words]
    total_text_tokens = sum(token_counts)

    # Proportional speech token assignment � same algorithm as
    # timed_tts.py lines 424-435. Each non-last word gets >= 1 token;
    # last word absorbs remainder (clamped to >= 0 to handle the edge
    # case where max(count,1) guards on earlier words overshoot).
    n_words = len(words)
    speech_allocs = []
    cursor = 0

    for i, tc in enumerate(token_counts):
        if i == n_words - 1:
            count = max(n_speech_tokens - cursor, 0)
        else:
            count = round(n_speech_tokens * tc / total_text_tokens)
            count = max(count, 1)
            remaining_after = n_words - 1 - i
            count = min(count, n_speech_tokens - cursor - remaining_after)
            count = max(count, 1)
        speech_allocs.append(count)
        cursor += count

    # Build timestamps
    timestamps = []
    t = time_offset

    for word, n_text_tok, n_speech_tok in zip(words, token_counts, speech_allocs):
        duration = n_speech_tok * SECS_PER_TOKEN
        timestamps.append(WordTimestamp(
            word=word,
            start=round(t, 4),
            end=round(t + duration, 4),
            text_tokens=n_text_tok,
            speech_tokens=n_speech_tok,
        ))
        t += duration

    return timestamps


def rescale_timestamps(
    raw_timestamps: List[WordTimestamp],
    natural_duration: float,
    actual_duration: float,
    absolute_offset: float = 0.0,
) -> List[dict]:
    """
    Rescale word timestamps from natural pace to actual (stretched) pace,
    then shift to an absolute position in the final audio.

    Returns List[dict] (JSON-serializable) rather than dataclasses,
    because these are the final output that flows into SegmentResult
    and JSON files � no further processing needed.
    """
    if not raw_timestamps or natural_duration <= 0:
        return []

    scale = actual_duration / natural_duration
    result = []
    for wt in raw_timestamps:
        result.append(dict(
            word=wt.word,
            start=round(absolute_offset + wt.start * scale, 4),
            end=round(absolute_offset + wt.end * scale, 4),
            text_tokens=wt.text_tokens,
            speech_tokens=wt.speech_tokens,
        ))
    return result


def timestamps_to_dict(timestamps: List[WordTimestamp]) -> List[dict]:
    """Convert to JSON-serializable list of dicts."""
    return [asdict(ts) for ts in timestamps]


def save_timestamps_json(
    timestamps: List[dict],
    path: str,
    extra_metadata: dict = None,
):
    """Save word timestamps to a JSON file."""
    data = {"words": timestamps}
    if extra_metadata:
        data.update(extra_metadata)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)