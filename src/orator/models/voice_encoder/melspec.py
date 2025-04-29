from functools import lru_cache

import numpy as np
import torch
from torchaudio.transforms import MelSpectrogram

from .config import VoiceEncConfig


class ResembleMelSpectrogram(torch.nn.Module):
    def __init__(self, hp=VoiceEncConfig()):
        """
        Torch implementation of Resemble's mel extraction.
        Note that the values are NOT identical to librosa's implementation due to floating point precisions, however
        the results are very very close. One test file gave an L1 error of just 0.005%, full results:
            Librosa mel max:  0.871768
            Torch mel max:    0.871768
            Librosa mel mean: 0.316302
            Torch mel mean:   0.316289
            Max diff:         0.061105
            Mean diff:        1.453384e-05
            Percent error:    0.004595%
        """
        super().__init__()
        self.melspec = MelSpectrogram(
            hp.sample_rate,
            n_fft=hp.n_fft,
            win_length=hp.win_size,
            hop_length=hp.hop_size,
            f_min=hp.fmin,
            f_max=hp.fmax,
            n_mels=hp.num_mels,
            power=1,
            normalized=False,
            # NOTE: Folowing librosa's default.
            pad_mode="constant",
            norm="slaney",
            mel_scale="slaney",
        )
        self.register_buffer(
            "stft_magnitude_min",
            torch.FloatTensor([hp.stft_magnitude_min])
        )
        self.min_level_db = 20 * np.log10(hp.stft_magnitude_min)
        self.preemphasis = hp.preemphasis
        self.hop_size = hp.hop_size

    def forward(self, wav, pad=True):
        """
        Args:
            wav: [B, T]
        """
        if self.preemphasis > 0:
            wav = torch.nn.functional.pad(wav, [1, 0], value=0)
            wav = wav[..., 1:] - self.preemphasis * wav[..., :-1]

        mel = self.melspec(wav)

        mel = self._amp_to_db(mel)
        mel_normed = self._normalize(mel)
        assert not pad or mel_normed.shape[-1] == 1 + \
            wav.shape[-1] // self.hop_size   # Sanity check
        return mel_normed   # (M, T)

    def _normalize(self, s, headroom_db=15):
        s = (s - self.min_level_db) / (-self.min_level_db + headroom_db)
        return s

    def _amp_to_db(self, x):
        return 20 * torch.maximum(self.stft_magnitude_min, x).log10()


@lru_cache()
def melspectrogram():
    return ResembleMelSpectrogram()
