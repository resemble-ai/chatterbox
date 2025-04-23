from functools import lru_cache

from scipy import signal
import numpy as np
import librosa
import torch


@lru_cache()
def mel_basis(hp):
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(
        sr=hp.sample_rate,
        n_fft=hp.n_fft,
        n_mels=hp.num_mels,
        fmin=hp.fmin,
        fmax=hp.fmax)  # -> (nmel, nfreq)


@lru_cache()
def inv_mel_basis(hp):
    return np.linalg.pinv(mel_basis(hp))  # (nfreq, nmel)


def preemphasis(wav, hp):
    assert hp.preemphasis != 0
    wav = signal.lfilter([1, -hp.preemphasis], [1], wav)
    wav = np.clip(wav, -1, 1)
    return wav


def inv_preemphasis(wav, hp):
    assert hp.preemphasis != 0
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)
    wav = np.clip(wav, -1, 1)
    return wav


def melspectrogram(wav, hp, pad=True):
    # Run through pre-emphasis
    if hp.preemphasis > 0:
        wav = preemphasis(wav, hp)
        assert np.abs(wav).max() - 1 < 1e-07

    # Do the stft
    spec_complex = _stft(wav, hp, pad=pad)

    # Get the magnitudes
    spec_magnitudes = np.abs(spec_complex)

    if hp.mel_power != 1.0:
        spec_magnitudes **= hp.mel_power

    # Get the mel and convert magnitudes->db
    mel = np.dot(mel_basis(hp), spec_magnitudes)
    if hp.mel_type == "db":
        mel = _amp_to_db(mel, hp)

    # Normalise the mel from db to 0,1
    if hp.normalized_mels:
        mel = _normalize(mel, hp).astype(np.float32)

    assert not pad or mel.shape[1] == 1 + len(wav) // hp.hop_size   # Sanity check
    return mel   # (M, T)


def complex_spectrogram(wav, hp):
    spec = _stft(wav, hp)
    return np.stack((spec.real, spec.imag), axis=-1)


def invert_mel(mel_spectrogram, hp, scaled=True, power=1.5, n_iters=60, headroom_db=15):
    if scaled:
        mel_spectrogram = unscale_spectrogram(mel_spectrogram, hp)

    min_level_db = 20 * np.log10(hp.stft_magnitude_min)
    D = (mel_spectrogram * (-min_level_db + headroom_db)) + min_level_db

    # Convert back to linear
    S = np.maximum(1e-10, np.dot(inv_mel_basis(hp), _db_to_amp(D)))

    wav = _griffin_lim(S ** power, n_iters, hp).astype(np.float32)

    if hp.preemphasis > 0:
        return inv_preemphasis(wav, hp)
    return wav


def _griffin_lim(S, n_iters, hp):
    """
    Based on https://github.com/librosa/librosa/issues/434
    """
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(complex)
    y = _istft(S_complex * angles, hp)
    for i in range(n_iters):
        angles = np.exp(1j * np.angle(_stft(y, hp)))
        y = _istft(S_complex * angles, hp)
    return y


def mel_sr_mask(true_sr, hp):
    """
    Builds a mel mask for audio that has a samplerate lower than the hparams (ie, was upsampled during preprocessing).
    Args:
        true_sr: shape (B,) tensor of true samplerates (sr of un-processed raw audio files)
        hp: HParams
    Returns:
        mask: shape (B, 1, num_mels) float tensor for masking mels
    """
    assert isinstance(true_sr, torch.Tensor) and true_sr.ndim == 1, "true_sr must be a batch tensor"
    B = len(true_sr)
    d = true_sr.device

    model_sr = hp.sample_rate
    true_sr[true_sr < 0] = model_sr  # negatives indicate unknown true SR

    sr_mask = torch.ones(B, 1, hp.num_mels).float().to(d)           # (B, 1, nmel)
    if (true_sr < model_sr).any():
        # convert true sr to linear spec index
        lin_idx = hp.num_freq * true_sr.float() / model_sr          # (B,)
        lin_idx = lin_idx.round().long().clamp(0, hp.num_freq - 1)  # (B,)

        # convert linear indices to mel indices
        mel_idx = convert_index(hp, linear_idx=lin_idx)             # (B,)
        mel_idx = mel_idx.to(d)[:, None]                            # (B, 1)

        # build mask
        arange = torch.arange(hp.num_mels, dtype=torch.long).to(d)[None, :]  # (1, nmel)
        sr_mask = (arange <= mel_idx)                                        # (B, nmel)
        sr_mask = sr_mask[:, None].float()                                   # (B, 1, nmel)

    return sr_mask  # (B, 1, nmel)


def convert_index(hp, mel_idx=None, linear_idx=None):
    "Convert a mel [linear] index into the corresponding linear [mel] index for the same frequency"
    assert (mel_idx is None) ^ (linear_idx is None), "provide exactly one index value"
    if mel_idx is None:
        basis, i = mel_basis(hp), linear_idx
    else:
        basis, i = inv_mel_basis(hp), mel_idx

    # ensure index is tensor and validate
    i = torch.as_tensor(i)
    is_scalar = (i.ndim == 0)
    i = torch.atleast_1d(i).long()  # (B,)
    assert i.ndim == 1, "too many dimensions for index, please flatten/reshape manually"
    d = i.device
    B = len(i)
    M, N = basis.shape

    # map through [inverse] mel basis
    basis = torch.as_tensor(basis).to(d)                # (M, N)
    query = torch.zeros(B, N).to(d)                     # (B, N)
    query.scatter_(1, i[:, None], 1)                    # (B, N)
    result = torch.einsum('mn,bn -> bm', basis, query)  # (B, M)

    # compute centroid index for each result vector
    norm = result / result.sum(dim=1, keepdim=True)      # (B, M)
    ar = torch.arange(M, dtype=torch.float)[None].to(d)  # (B,)
    centroid = (norm * ar).sum(dim=1)                    # (B,)
    centroid = centroid.round().long()                   # (B,)

    # output formatting
    if is_scalar:
        centroid = centroid.squeeze(0)
    return centroid


def _stft(y, hp, pad=True):
    # NOTE: after 0.8, pad mode defaults to constant, setting this to reflect for
    #   historical consistency and streaming-version consistency
    return librosa.stft(
        y,
        n_fft=hp.n_fft,
        hop_length=hp.hop_size,
        win_length=hp.win_size,
        center=pad,
        pad_mode="reflect",
    )


def _istft(y, hp):
    return librosa.istft(y, hop_length=hp.hop_size, win_length=hp.win_size)


def _amp_to_db(x, hp):
    return 20 * np.log10(np.maximum(hp.stft_magnitude_min, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(s, hp, headroom_db=15):
    min_level_db = 20 * np.log10(hp.stft_magnitude_min)
    s = (s - min_level_db) / (-min_level_db + headroom_db)
    return s


def scale_spectrogram(x, hp):
    if hp.syn_symmetric_mel:
        x = (x - 0.5) * 2 * hp.syn_mel_scale
    else:
        x = x * hp.syn_mel_scale
    return x


def unscale_spectrogram(x, hp):
    if hp.syn_symmetric_mel:
        x = (x + hp.syn_mel_scale) / (2 * hp.syn_mel_scale)
    else:
        x = x / hp.syn_mel_scale

    if torch.is_tensor(x):
        x = torch.clamp(x, 0, 1)
    else:
        x = np.clip(x, 0, 1)

    return x


class PytorchSpectrogram(torch.nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.n_fft = hp.n_fft
        self.hop_size = hp.hop_size
        self.win_size = hp.win_size

        # TODO: make window configurable?
        self.register_buffer("window", torch.hann_window(hp.win_size))

    def forward(self, wav, center=False):
        """
        Added for Sovits model.
        Adapted from [So-VITS-SVC-fork](https://github.com/voicepaw/so-vits-svc-fork/blob/main/src/so_vits_svc_fork/modules/mel_processing.py)
        which was adapted from [the official VITS repo](https://github.com/jaywalnut310/vits/blob/main/mel_processing.py)

        Args:
            y: [(B,) T] waveform
            hp: HParams

        Returns:
            spec: spectrogram in linear frequency-scale & linear magnitude.
        """
        if input_is_np := isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).unsqueeze(0)  # -> [1, T]

        wav = torch.nn.functional.pad(
            ## So-VITS-SVC's setting results in 1 frame difference
            # wav.unsqueeze(1),
            # (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),

            # The following results in spec well-aligned with Resemble's.
            wav,
            [self.n_fft // 2, self.n_fft // 2],
            mode="reflect",
        )

        spec = torch.stft(
            wav,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window,
            # TODO: Make `center` and `pad_mode` configurable?
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
        if input_is_np:
            spec = spec.squeeze(0).numpy()
        return spec
