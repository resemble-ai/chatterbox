"""Utility helpers for loading and normalising audio inputs."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np
import librosa


AudioLike = Union[str, Path]


def _coerce_to_paths(audio: Union[AudioLike, Sequence[AudioLike]]) -> Sequence[Path]:
    """Convert an input that may be a single path or a list of paths into ``Path`` objects."""
    if isinstance(audio, (str, Path)):
        return [Path(audio)]

    if isinstance(audio, Sequence):
        if not audio:
            raise ValueError("Expected at least one reference audio file, received an empty sequence.")
        coerced: list[Path] = []
        for item in audio:
            if not isinstance(item, (str, Path)):
                raise TypeError(
                    "Reference audio entries must be file paths (str or Path); "
                    f"received unsupported type: {type(item)!r}"
                )
            coerced.append(Path(item))
        return coerced

    raise TypeError(
        "Reference audio must be provided as a single path or a sequence of paths. "
        f"Received unsupported type: {type(audio)!r}"
    )


def trim_silence(wav: np.ndarray, top_db: float = 40.0) -> np.ndarray:
    """Remove leading and trailing silence from ``wav`` using an energy threshold."""
    trimmed, index = librosa.effects.trim(wav, top_db=top_db)
    if index is None or trimmed.size == 0:
        return wav
    return trimmed


def loudness_normalise(wav: np.ndarray, target_db: float = -27.0, eps: float = 1e-7) -> np.ndarray:
    """Normalise the RMS loudness of ``wav`` toward ``target_db`` while avoiding clipping."""
    rms = float(np.sqrt(np.mean(np.square(wav)) + eps))
    if rms <= eps:
        return wav

    current_db = 20.0 * np.log10(rms + eps)
    gain = 10.0 ** ((target_db - current_db) / 20.0)
    normalised = wav * gain
    peak = np.max(np.abs(normalised))
    if peak > 1.0:
        normalised = normalised / peak
    return normalised.astype(np.float32)


def load_and_condition_reference(
    audio: Union[AudioLike, Sequence[AudioLike]],
    target_sr: int,
    max_samples: int,
    top_db: float = 40.0,
    target_loudness_db: float = -27.0,
) -> np.ndarray:
    """
    Load one or more reference audio files, trim silence, perform loudness normalisation and
    concatenate them (in the provided order).

    The combined waveform is truncated to ``max_samples``.
    """
    paths = _coerce_to_paths(audio)

    processed: list[np.ndarray] = []
    for path in paths:
        wav, _ = librosa.load(path.as_posix(), sr=target_sr)
        wav = trim_silence(wav, top_db=top_db)
        wav = loudness_normalise(wav, target_db=target_loudness_db)
        if wav.size == 0:
            raise ValueError(f"Reference audio '{path}' appears to be silent after trimming.")
        processed.append(wav.astype(np.float32))

    combined = np.concatenate(processed)
    if combined.size > max_samples:
        combined = combined[:max_samples]
    return combined


def load_source_audio(
    audio_path: AudioLike,
    target_sr: int,
    top_db: float | None = None,
    normalise: bool = False,
) -> np.ndarray:
    """Load a source waveform and optionally trim and normalise it."""
    wav, _ = librosa.load(Path(audio_path).as_posix(), sr=target_sr)
    if top_db is not None:
        wav = trim_silence(wav, top_db=top_db)
    if normalise:
        wav = loudness_normalise(wav)
    return wav.astype(np.float32)
