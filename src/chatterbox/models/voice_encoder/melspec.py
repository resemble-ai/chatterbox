from functools import lru_cache

from scipy import signal
import numpy as np
import torchaudio.functional as taF
import torch


@lru_cache()
def mel_basis(hp):
    assert hp.fmax <= hp.sample_rate // 2
    n_freqs = hp.n_fft // 2 + 1
    fb = taF.melscale_fbanks(
        n_freqs=n_freqs,
        f_min=float(hp.fmin),
        f_max=float(hp.fmax),
        n_mels=int(hp.num_mels),
        sample_rate=int(hp.sample_rate),
        norm="slaney",
        mel_scale="slaney",
    )  # (n_freqs, n_mels)
    return fb.T  # -> (nmel, nfreq) as tensor


def preemphasis(wav, hp):
    assert hp.preemphasis != 0
    wav = signal.lfilter([1, -hp.preemphasis], [1], wav)
    wav = np.clip(wav, -1, 1)
    return wav


def melspectrogram(wav, hp, pad=True):
    # Run through pre-emphasis
    if hp.preemphasis > 0:
        wav = preemphasis(wav, hp)
        assert np.abs(wav).max() - 1 < 1e-07

    # Do the stft using torch for performance
    spec_complex = _stft(wav, hp, pad=pad)

    # Get the magnitudes
    spec_magnitudes = torch.abs(spec_complex)

    if hp.mel_power != 1.0:
        spec_magnitudes **= hp.mel_power

    # Get the mel and convert magnitudes->db
    mel_filters = mel_basis(hp).to(spec_magnitudes.device)  # Ensure same device
    mel = mel_filters @ spec_magnitudes
    if hp.mel_type == "db":
        mel = _amp_to_db_tensor(mel, hp)

    # Normalise the mel from db to 0,1
    if hp.normalized_mels:
        mel = _normalize_tensor(mel, hp).to(torch.float32)

    assert not pad or mel.shape[1] == 1 + len(wav) // hp.hop_size   # Sanity check
    return mel   # (M, T) as tensor


def _stft(y, hp, pad=True):
    # Convert to torch tensor and compute STFT; keep as tensor
    if not torch.is_tensor(y):
        y_t = torch.from_numpy(y.astype(np.float32))
    else:
        y_t = y
    stft = torch.stft(
        y_t,
        n_fft=hp.n_fft,
        hop_length=hp.hop_size,
        win_length=hp.win_size,
        center=pad,
        window=torch.hann_window(hp.win_size, device=y_t.device),
        pad_mode="reflect",
        return_complex=True,
    )
    return stft  # Keep as tensor


def _amp_to_db(x, hp):
    return 20 * np.log10(np.maximum(hp.stft_magnitude_min, x))


def _amp_to_db_tensor(x, hp):
    return 20 * torch.log10(torch.maximum(torch.tensor(hp.stft_magnitude_min, device=x.device, dtype=x.dtype), x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(s, hp, headroom_db=15):
    min_level_db = 20 * np.log10(hp.stft_magnitude_min)
    s = (s - min_level_db) / (-min_level_db + headroom_db)
    return s


def _normalize_tensor(s, hp, headroom_db=15):
    min_level_db = 20 * np.log10(hp.stft_magnitude_min)
    s = (s - min_level_db) / (-min_level_db + headroom_db)
    return s
