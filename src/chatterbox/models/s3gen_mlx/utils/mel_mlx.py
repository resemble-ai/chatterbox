# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
MLX utility functions for mel-spectrogram extraction.
Note: For full mel extraction we'll need to interface with PyTorch/librosa
since MLX doesn't have native STFT support.
"""

import mlx.core as mx
import numpy as np
from typing import Optional


def dynamic_range_compression(
    x: mx.array, C: float = 1.0, clip_val: float = 1e-5
) -> mx.array:
    """Apply dynamic range compression (log compression).

    Args:
        x: Input magnitude spectrogram
        C: Compression constant
        clip_val: Minimum value for clipping

    Returns:
        Compressed spectrogram
    """
    return mx.log(mx.clip(x, a_min=clip_val, a_max=None) * C)


def spectral_normalize(magnitudes: mx.array) -> mx.array:
    """Normalize spectrogram using dynamic range compression."""
    return dynamic_range_compression(magnitudes)


def mel_spectrogram_mlx(
    audio: mx.array,
    n_fft: int = 1024,
    num_mels: int = 80,
    sampling_rate: int = 24000,
    hop_size: int = 256,
    win_size: int = 1024,
    fmin: int = 0,
    fmax: Optional[int] = 8000,
) -> mx.array:
    """
    Compute mel spectrogram from audio.

    Note: This is a placeholder that requires PyTorch/librosa for STFT.
    For inference, use the PyTorch mel extraction and convert to MLX.

    Args:
        audio: Input audio waveform [B, T] or [T]
        n_fft: FFT size
        num_mels: Number of mel bands
        sampling_rate: Audio sample rate
        hop_size: Hop size for STFT
        win_size: Window size for STFT
        fmin: Minimum frequency for mel
        fmax: Maximum frequency for mel

    Returns:
        Mel spectrogram [B, num_mels, T_mel]
    """
    # MLX doesn't have native STFT support, so we need to use numpy/scipy
    # For production, pre-compute mels with PyTorch and pass them directly
    import warnings

    warnings.warn(
        "mel_spectrogram_mlx requires scipy for STFT. "
        "For best performance, pre-compute mels using PyTorch.",
        UserWarning,
    )

    try:
        import scipy.signal
        import scipy.fft

        # Convert to numpy
        if hasattr(audio, "tolist"):
            audio_np = np.array(audio.tolist())
        else:
            audio_np = np.array(audio)

        # Ensure 2D [B, T]
        if len(audio_np.shape) == 1:
            audio_np = audio_np[np.newaxis, :]

        batch_size = audio_np.shape[0]

        # Create mel filterbank
        mel_basis = _mel_filterbank(
            sampling_rate,
            n_fft,
            num_mels,
            fmin,
            fmax if fmax is not None else sampling_rate // 2,
        )

        # Hann window
        window = scipy.signal.windows.hann(win_size, sym=False)

        mels = []
        for i in range(batch_size):
            # STFT
            _, _, Zxx = scipy.signal.stft(
                audio_np[i],
                fs=sampling_rate,
                window=window,
                nperseg=win_size,
                noverlap=win_size - hop_size,
                nfft=n_fft,
                boundary=None,
                padded=False,
            )

            # Magnitude spectrogram
            mag = np.abs(Zxx)

            # Apply mel filterbank
            mel = np.dot(mel_basis, mag)

            # Log compression
            mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))

            mels.append(mel)

        # Stack and convert to MLX
        mels = np.stack(mels, axis=0)
        return mx.array(mels)

    except ImportError:
        raise RuntimeError(
            "mel_spectrogram_mlx requires scipy. " "Install with: pip install scipy"
        )


def _mel_filterbank(
    sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float
) -> np.ndarray:
    """Create mel filterbank matrix.

    Args:
        sr: Sample rate
        n_fft: FFT size
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency

    Returns:
        Mel filterbank matrix [n_mels, n_fft // 2 + 1]
    """

    # Mel scale conversion
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    # Mel points
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # FFT bin points
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    # Create filterbank
    n_freqs = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_freqs))

    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        # Rising slope
        for j in range(left, center):
            if center != left:
                filterbank[i, j] = (j - left) / (center - left)

        # Falling slope
        for j in range(center, right):
            if right != center:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank


# Note: Full mel_spectrogram extraction requires STFT which MLX doesn't support natively.
# For inference, mel spectrograms are computed using the PyTorch version in s3gen/utils/mel.py
# and then converted to MLX arrays for further processing.
