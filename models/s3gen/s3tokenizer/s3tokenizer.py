from typing import List, Dict, Tuple, Union, Optional

import numpy as np
import librosa
import torch
import torch.nn.functional as F
from s3tokenizer.utils import padding
from s3tokenizer.model_v2 import (
    S3TokenizerV2,
    ModelConfig,
)
from accelerate import Accelerator


from models.s3gen.s3tokenizer.const import S3_SR, S3_HOP


class S3Tokenizer(S3TokenizerV2):
    """
    A copy of s3tokenizer.S3TokenizerV2 with the following changes:
    - a more integrated `forward`
    - compute `log_mel_spectrogram` using `_mel_filters` and `window` in `register_buffers`
    """

    ignore_state_dict_missing = ("_mel_filters", "window")

    def __init__(
        self,
        name: str="speech_tokenizer_v2_25hz",
        config: ModelConfig = ModelConfig()
    ):
        super().__init__(name)

        self.n_fft = 400
        _mel_filters = librosa.filters.mel(
            sr=S3_SR,
            n_fft=self.n_fft,
            n_mels=config.n_mels
        )
        self.register_buffer(
            "_mel_filters",
            torch.FloatTensor(_mel_filters),
        )

        self.register_buffer(
            "window",
            torch.hann_window(self.n_fft),
        )

    def _prepare_audio(self, wavs, autopad=False):
        """Prepare a list of audios for s3tokenizer processing."""
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            # NOTE: Pad 4 mel frames (1 speech token) to reduce the wave len
            # difference between input and output decoded from the Cosyvoice2 decoder.
            if autopad:
                n_mels = wav.shape[-1] // S3_HOP + 1
                n_mel_pad = 8
                n_pad = (n_mels + n_mel_pad) * S3_HOP  - wav.shape[-1]
                wav = F.pad(wav, (0, n_pad))
            processed_wavs.append(wav)
        return processed_wavs

    @torch.no_grad()
    def forward(
        self,
        wavs: torch.Tensor,
        accelerator: Accelerator=None,
        max_len: int=None,
        autopad=True,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        NOTE: mel-spec has a hop size of 160 points (100 frame/sec).
        FIXME: this class inherits `nn.Module` but doesn't accept `torch.Tensor` and handles a list of wavs one by one, which is unexpected.

        Args
        ----
        - `wavs`: 16 kHz speech audio
        - `max_len` max length to truncate the output sequence to (25 token/sec).
        NOTE: please pad the waveform if longer sequence is needed.
        """
        processed_wavs = self._prepare_audio(wavs, autopad)
        mels, mel_lens = [], []
        for wav in processed_wavs:
            wav = wav.to(self.device)
            mel = self.log_mel_spectrogram(wav)  # [B=1, F, T]
            if max_len is not None:
                mel = mel[..., :max_len * 4]  # num_mel_frames = 4 * num_tokens
            mels.append(mel.squeeze(0))

        mels, mel_lens = padding(mels)
        if accelerator is None:
            tokenizer = self
        else:
            tokenizer = accelerator.unwrap_model(self)

        speech_tokens, speech_token_lens = tokenizer.quantize(mels, mel_lens.to(self.device))
        return (
            speech_tokens.long().detach(),
            speech_token_lens.long().detach(),
        )

    def log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        padding: int = 0,
    ):
        """
        Compute the log-Mel spectrogram of

        Parameters
        ----------
        audio: torch.Tensor, shape = (*)
            The path to audio or either a NumPy array or Tensor containing the
            audio waveform in 16 kHz

        padding: int
            Number of zero samples to pad to the right

        Returns
        -------
        torch.Tensor, shape = (128, n_frames)
            A Tensor that contains the Mel spectrogram
        """
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)

        audio = audio.to(self.device)
        if padding > 0:
            audio = F.pad(audio, (0, padding))
        stft = torch.stft(
            audio, self.n_fft, S3_HOP,
            window=self.window.to(self.device),
            return_complex=True
        )
        magnitudes = stft[..., :-1].abs()**2

        mel_spec = self._mel_filters.to(self.device) @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec


def test():
    from pathlib import Path
    import librosa
    import s3tokenizer
    from data_objects.models.checkpoint_manager import CheckpointManager

    orig = s3tokenizer.load_model("speech_tokenizer_v2_25hz")

    ours_cm = CheckpointManager(
        Path("saved_models"),
        "s3tokenizer/v2_25hz",
        model_type="s3tokenizer",
        check_cloud=True,
        load_only=True,
        bucket_name="resemble-model-files"
    )
    # ours = S3Tokenizer()
    # ours.load_state_dict(orig.state_dict(), strict=False)
    # ours_cm.save(ours, 100_000)
    ours = S3Tokenizer.instantiate_from_cm( ours_cm, "cpu", mode="inference", return_step=False, state_dict_mismatch="raise" )

    wav1, sr = librosa.load("/mnt/tts-en-augment/datasets/anispeech_32k_pp/wavs/32_15.wav")
    # wav2, sr = librosa.load("/mnt/tts-en-augment/datasets/anispeech_32k_pp/wavs/32_35.wav")

    # wavs = [wav1, wav2]
    # x = tokenizer(wavs)
    xm, _ = ours([wav1])
    xp, _ = ours([wav1], autopad=True)


    mel = s3tokenizer.log_mel_spectrogram(wav1)
    mels, mel_lens = padding([mel])
    xo, ls = orig(mels, mel_lens)
    if 0 == (xo - xm).sum():
        print("\n✅Identical results\n")
    else:
        raise ValueError("\n⚠️Results differ\n")

    print("Note that if the input is padded, there will be slight changes in the outputs.")
    print(xp[0, : xm.shape[1]] - xm[0])


if __name__ == "__main__":
    test()
