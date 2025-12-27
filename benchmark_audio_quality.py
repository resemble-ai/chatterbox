#!/usr/bin/env python3
"""
Audio Quality Validation Module for TTS Benchmarks
==================================================

Provides comprehensive audio quality checks:
- Duration validation (expected vs actual)
- Transcription accuracy (using mlx_whisper)
- Word Error Rate (WER)
- Truncation detection (missing first/last words)
- Audio fidelity metrics
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import librosa
import numpy as np


@dataclass
class AudioQualityMetrics:
    """Comprehensive audio quality metrics."""
    # Duration metrics
    expected_duration_s: float
    actual_duration_s: float
    duration_error_s: float
    duration_error_pct: float

    # Transcription quality
    transcription: Optional[str] = None
    word_error_rate: Optional[float] = None

    # Content validation
    has_first_word: bool = False
    has_last_word: bool = False
    first_word_expected: Optional[str] = None
    last_word_expected: Optional[str] = None
    first_word_transcribed: Optional[str] = None
    last_word_transcribed: Optional[str] = None

    # Truncation detection
    is_truncated: bool = False
    truncation_reason: Optional[str] = None

    # Audio fidelity
    has_hard_cutoff: bool = False
    trailing_silence_s: float = 0.0
    leading_silence_s: float = 0.0

    # Overall quality score
    quality_score: float = 0.0  # 0-100, higher is better
    quality_status: str = "unknown"  # "excellent", "good", "fair", "poor"


def calculate_word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER) between reference and hypothesis.

    Args:
        reference: Reference text
        hypothesis: Hypothesis text from transcription

    Returns:
        WER as a float (0.0 = perfect match, 1.0 = all errors)
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # Dynamic programming for Levenshtein distance
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32)

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j] + 1,    # deletion
                             d[i][j-1] + 1,    # insertion
                             d[i-1][j-1] + 1)  # substitution

    wer = d[len(ref_words)][len(hyp_words)] / max(len(ref_words), 1)
    return wer


def transcribe_audio(audio_path: str, language: str = "en") -> Optional[str]:
    """
    Transcribe audio using MLX Whisper.

    Args:
        audio_path: Path to audio file
        language: Language code for transcription hint

    Returns:
        Transcription text or None if failed
    """
    try:
        import mlx_whisper
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo="mlx-community/whisper-base-mlx-q4",
            language=language if language in ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "zh", "ko"] else None,
            verbose=False,
        )
        return result["text"].strip()
    except ImportError:
        print("⚠️  mlx_whisper not installed. Install with: pip install mlx-whisper")
        return None
    except Exception as e:
        print(f"⚠️  Transcription failed: {e}")
        return None


def detect_truncation(
    audio_path: str,
    reference_text: str,
    transcription: Optional[str] = None,
) -> Tuple[bool, Optional[str], bool, bool]:
    """
    Detect if audio has been truncated.

    Args:
        audio_path: Path to audio file
        reference_text: Expected text content
        transcription: Optional pre-computed transcription

    Returns:
        Tuple of (is_truncated, reason, has_first_word, has_last_word)
    """
    # Transcribe if not provided
    if transcription is None:
        transcription = ""

    # Get expected first and last words
    ref_words = reference_text.strip().split()
    if len(ref_words) == 0:
        return False, None, False, False

    first_word_expected = ref_words[0].lower().strip('.,!?;:"\'-')
    last_word_expected = ref_words[-1].lower().strip('.,!?;:"\'-')

    # Get transcribed first and last words
    trans_words = transcription.lower().split()
    has_first_word = False
    has_last_word = False

    if len(trans_words) > 0:
        first_word_transcribed = trans_words[0].strip('.,!?;:"\'-')
        last_word_transcribed = trans_words[-1].strip('.,!?;:"\'-')

        # Fuzzy match (allows for minor transcription errors)
        has_first_word = (
            first_word_expected in first_word_transcribed or
            first_word_transcribed in first_word_expected or
            first_word_expected[:3] == first_word_transcribed[:3]  # First 3 chars match
        )

        has_last_word = (
            last_word_expected in last_word_transcribed or
            last_word_transcribed in last_word_expected or
            last_word_expected[:3] == last_word_transcribed[:3]  # First 3 chars match
        )

    # Determine truncation
    is_truncated = False
    reason = None

    if not has_first_word:
        is_truncated = True
        reason = f"Missing first word: '{first_word_expected}'"
    elif not has_last_word:
        is_truncated = True
        reason = f"Missing last word: '{last_word_expected}'"

    return is_truncated, reason, has_first_word, has_last_word


def check_hard_cutoff(audio_path: str, threshold: float = 0.01) -> Tuple[bool, float, float]:
    """
    Check if audio has a hard cutoff (abrupt ending without fade).

    Args:
        audio_path: Path to audio file
        threshold: Amplitude threshold for silence detection

    Returns:
        Tuple of (has_hard_cutoff, trailing_silence_s, leading_silence_s)
    """
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=24000)

        if len(audio) == 0:
            return True, 0.0, 0.0

        # Check last 10 samples for hard cutoff
        last_samples = audio[-10:]
        max_amplitude = np.max(np.abs(last_samples))
        has_hard_cutoff = max_amplitude > threshold

        # Calculate trailing silence
        frame_length = sr // 20  # 50ms frames
        trailing_silence_frames = 0

        for i in range(len(audio) - frame_length, 0, -frame_length):
            frame = audio[i:i+frame_length]
            if len(frame) == 0:
                break
            rms = np.sqrt(np.mean(frame**2))
            if rms < threshold:
                trailing_silence_frames += 1
            else:
                break

        trailing_silence_s = (trailing_silence_frames * frame_length) / sr

        # Calculate leading silence
        leading_silence_frames = 0
        for i in range(0, len(audio), frame_length):
            frame = audio[i:i+frame_length]
            if len(frame) == 0:
                break
            rms = np.sqrt(np.mean(frame**2))
            if rms < threshold:
                leading_silence_frames += 1
            else:
                break

        leading_silence_s = (leading_silence_frames * frame_length) / sr

        return has_hard_cutoff, trailing_silence_s, leading_silence_s

    except Exception as e:
        print(f"⚠️  Hard cutoff check failed: {e}")
        return False, 0.0, 0.0


def validate_audio_quality(
    audio_path: str,
    reference_text: str,
    language: str = "en",
    enable_transcription: bool = True,
) -> AudioQualityMetrics:
    """
    Comprehensive audio quality validation.

    Args:
        audio_path: Path to generated audio file
        reference_text: Expected text content
        language: Language code for transcription
        enable_transcription: Whether to perform transcription validation

    Returns:
        AudioQualityMetrics with comprehensive quality information
    """
    # Calculate expected duration (~0.5s per word)
    num_words = len(reference_text.split())
    expected_duration = num_words * 0.5

    # Get actual duration
    try:
        audio, sr = librosa.load(audio_path, sr=24000)
        actual_duration = len(audio) / sr
    except Exception as e:
        print(f"⚠️  Failed to load audio for quality check: {e}")
        return AudioQualityMetrics(
            expected_duration_s=expected_duration,
            actual_duration_s=0.0,
            duration_error_s=0.0,
            duration_error_pct=100.0,
            quality_score=0.0,
            quality_status="error",
        )

    duration_error_s = actual_duration - expected_duration
    duration_error_pct = (duration_error_s / expected_duration) * 100 if expected_duration > 0 else 0

    # Transcription validation
    transcription = None
    wer = None
    first_word_transcribed = None
    last_word_transcribed = None

    if enable_transcription:
        transcription = transcribe_audio(audio_path, language)
        if transcription:
            wer = calculate_word_error_rate(reference_text, transcription)
            trans_words = transcription.split()
            if len(trans_words) > 0:
                first_word_transcribed = trans_words[0].strip('.,!?;:"\'-')
                last_word_transcribed = trans_words[-1].strip('.,!?;:"\'-')

    # Truncation detection
    is_truncated, truncation_reason, has_first_word, has_last_word = detect_truncation(
        audio_path, reference_text, transcription
    )

    # Hard cutoff detection
    has_hard_cutoff, trailing_silence_s, leading_silence_s = check_hard_cutoff(audio_path)

    # Extract expected first/last words
    ref_words = reference_text.split()
    first_word_expected = ref_words[0].strip('.,!?;:"\'-') if len(ref_words) > 0 else None
    last_word_expected = ref_words[-1].strip('.,!?;:"\'-') if len(ref_words) > 0 else None

    # Calculate quality score (0-100)
    quality_score = 100.0

    # Duration penalty (-10 points per 10% error)
    duration_penalty = min(abs(duration_error_pct) / 10 * 10, 30)
    quality_score -= duration_penalty

    # WER penalty (-40 points for WER=1.0)
    if wer is not None:
        wer_penalty = wer * 40
        quality_score -= wer_penalty

    # Truncation penalty
    if is_truncated:
        quality_score -= 20

    # Hard cutoff penalty
    if has_hard_cutoff:
        quality_score -= 10

    quality_score = max(0, quality_score)

    # Determine quality status
    if quality_score >= 90:
        quality_status = "excellent"
    elif quality_score >= 75:
        quality_status = "good"
    elif quality_score >= 60:
        quality_status = "fair"
    else:
        quality_status = "poor"

    return AudioQualityMetrics(
        expected_duration_s=expected_duration,
        actual_duration_s=actual_duration,
        duration_error_s=duration_error_s,
        duration_error_pct=duration_error_pct,
        transcription=transcription,
        word_error_rate=wer,
        has_first_word=has_first_word,
        has_last_word=has_last_word,
        first_word_expected=first_word_expected,
        last_word_expected=last_word_expected,
        first_word_transcribed=first_word_transcribed,
        last_word_transcribed=last_word_transcribed,
        is_truncated=is_truncated,
        truncation_reason=truncation_reason,
        has_hard_cutoff=has_hard_cutoff,
        trailing_silence_s=trailing_silence_s,
        leading_silence_s=leading_silence_s,
        quality_score=quality_score,
        quality_status=quality_status,
    )


def print_quality_report(metrics: AudioQualityMetrics, verbose: bool = True, reference_text: Optional[str] = None):
    """
    Print a formatted quality report.

    Args:
        metrics: AudioQualityMetrics to report
        verbose: Whether to print detailed information
        reference_text: Optional reference text to display
    """
    # Overall status with emoji
    status_emoji = {
        "excellent": "✅",
        "good": "✓",
        "fair": "⚠️",
        "poor": "❌",
        "error": "💥",
        "unknown": "❓",
    }

    emoji = status_emoji.get(metrics.quality_status, "❓")
    print(f"\n  {emoji} Quality: {metrics.quality_status.upper()} (score: {metrics.quality_score:.1f}/100)")

    if not verbose:
        return

    # Duration
    print(f"  Duration: {metrics.actual_duration_s:.2f}s (expected: {metrics.expected_duration_s:.1f}s, "
          f"error: {metrics.duration_error_s:+.2f}s / {metrics.duration_error_pct:+.1f}%)")

    # Truncation
    if metrics.is_truncated:
        print(f"  ⚠️  TRUNCATED: {metrics.truncation_reason}")
        if not metrics.has_first_word:
            print(f"      Missing first word: '{metrics.first_word_expected}'")
        if not metrics.has_last_word:
            print(f"      Missing last word: '{metrics.last_word_expected}'")

    # Transcription
    if metrics.transcription:
        if metrics.word_error_rate is not None:
            wer_pct = metrics.word_error_rate * 100
            print(f"  WER: {wer_pct:.1f}%")

        # Show first/last word match
        if verbose:
            first_match = "✓" if metrics.has_first_word else "✗"
            last_match = "✓" if metrics.has_last_word else "✗"
            print(f"  First word: {first_match} '{metrics.first_word_expected}' → '{metrics.first_word_transcribed}'")
            print(f"  Last word:  {last_match} '{metrics.last_word_expected}' → '{metrics.last_word_transcribed}'")

        # Show expected and transcribed text
        if reference_text:
            print(f"\n  Expected text:")
            print(f"    {reference_text}")
        print(f"\n  Transcript:")
        print(f"    {metrics.transcription}")

    # Audio fidelity
    if metrics.has_hard_cutoff:
        print(f"  ⚠️  Hard cutoff detected (no fade to silence)")
    if metrics.trailing_silence_s > 0.5:
        print(f"  Trailing silence: {metrics.trailing_silence_s:.2f}s")
