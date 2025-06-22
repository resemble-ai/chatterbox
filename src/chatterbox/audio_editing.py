import numpy as np
import librosa


def splice_audios(audios):
    """Concatenate a list of 1D audio arrays."""
    if not audios:
        raise ValueError("No audio segments provided")
    return np.concatenate(audios)


def trim_audio(audio, start_sec=0.0, end_sec=None, sr=22050):
    """Trim a section from ``start_sec`` to ``end_sec`` (in seconds)."""
    start = max(0, int(start_sec * sr))
    end = len(audio) if end_sec is None else int(end_sec * sr)
    if start >= end:
        return np.array([], dtype=audio.dtype)
    return audio[start:end]


def insert_audio(base_audio, insert_audio, position_sec, sr=22050):
    """Insert ``insert_audio`` into ``base_audio`` at ``position_sec`` seconds."""
    pos = max(0, min(len(base_audio), int(position_sec * sr)))
    return np.concatenate([base_audio[:pos], insert_audio, base_audio[pos:]])


def delete_segment(audio, start_sec, end_sec, sr=22050):
    """Remove the section between ``start_sec`` and ``end_sec`` seconds."""
    start = max(0, int(start_sec * sr))
    end = min(len(audio), int(end_sec * sr))
    return np.concatenate([audio[:start], audio[end:]])


def crossfade(audio1, audio2, duration_sec=0.01, sr=22050):
    """Crossfade ``audio1`` into ``audio2`` over ``duration_sec`` seconds."""
    n = int(duration_sec * sr)
    if n <= 0:
        return np.concatenate([audio1, audio2])
    fade_out = np.linspace(1.0, 0.0, n)
    fade_in = np.linspace(0.0, 1.0, n)
    a1_end = audio1[-n:] * fade_out
    a2_start = audio2[:n] * fade_in
    mixed = a1_end + a2_start
    return np.concatenate([audio1[:-n], mixed, audio2[n:]])

