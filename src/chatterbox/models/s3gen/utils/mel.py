"""mel-spectrogram extraction in Matcha-TTS"""
# Replaced librosa with torchaudio to avoid CPU transfers
import torch
import numpy as np
import torchaudio.functional as taF


# NOTE: they decalred these global vars
mel_basis = {}
hann_window = {}


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

"""
feat_extractor: !name:matcha.utils.audio.mel_spectrogram
    n_fft: 1920
    num_mels: 80
    sampling_rate: 24000
    hop_size: 480
    win_size: 1920
    fmin: 0
    fmax: 8000
    center: False

"""

def mel_spectrogram(y, n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480, win_size=1920,
                    fmin=0, fmax=8000, center=False):
    """Copied from https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/utils/audio.py
    Set default values according to Cosyvoice's config.
    """

    if isinstance(y, np.ndarray):
        y = torch.tensor(y).float()

    if len(y.shape) == 1:
        y = y[None, ]

    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window  # pylint: disable=global-statement,global-variable-not-assigned
    key = f"{fmax}_{y.device}"
    if key not in mel_basis:
        n_freqs = n_fft // 2 + 1
        # torchaudio returns (n_freqs, n_mels); we need (n_mels, n_freqs)
        mel = taF.melscale_fbanks(
            n_freqs=n_freqs,
            f_min=float(fmin),
            f_max=float(fmax),
            n_mels=int(num_mels),
            sample_rate=int(sampling_rate),
            norm="slaney",
            mel_scale="slaney",
        ).T
        mel_basis[key] = mel.to(dtype=torch.float32, device=y.device)
    
    # Create cache key that includes both win_size and device to avoid collisions
    win_key = f"{win_size}_{y.device}"
    if win_key not in hann_window:
        hann_window[win_key] = torch.hann_window(win_size, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec_c = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[win_key],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    spec = torch.abs(spec_c)

    spec = torch.matmul(mel_basis[key], spec)
    spec = spectral_normalize_torch(spec)

    return spec
