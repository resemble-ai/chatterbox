import logging
import torch
import torchaudio as ta
import numpy as np

logger = logging.getLogger(__name__)

_fb = ta.functional.melscale_fbanks if hasattr(ta.functional, "melscale_fbanks") else ta.functional.create_fb_matrix

mel_basis = {}
hann_window = {}


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def mel_spectrogram(y, n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480, win_size=1920,
                    fmin=0, fmax=8000, center=False):
    if isinstance(y, np.ndarray):
        y = torch.tensor(y).float()

    if len(y.shape) == 1:
        y = y[None, ]

    min_val = torch.min(y)
    max_val = torch.max(y)
    if min_val < -1.0 or max_val > 1.0:
        logger.warning(f"Audio values outside normalized range: min={min_val.item():.4f}, max={max_val.item():.4f}")

    global mel_basis, hann_window
    key = f"{fmax}_{y.device}"
    if key not in mel_basis:
        fb = _fb(
            n_freqs=n_fft // 2 + 1,
            n_mels=num_mels,
            sample_rate=sampling_rate,
            f_min=fmin,
            f_max=fmax,
        ).T
        mel_basis[key] = fb.to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[key], spec)
    spec = spectral_normalize_torch(spec)

    return spec