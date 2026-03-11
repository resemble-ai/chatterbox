# Adapted from https://github.com/CorentinJ/Real-Time-Voice-Cloning
# MIT License
from typing import List, Union, Optional

import numpy as np
from numpy.lib.stride_tricks import as_strided
import librosa
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .config import VoiceEncConfig
from .melspec import melspectrogram


def pack(arrays, seq_len: int=None, pad_value=0):
    """
    Given a list of length B of array-like objects of shapes (Ti, ...), packs them in a single tensor of
    shape (B, T, ...) by padding each individual array on the right.

    :param arrays: a list of array-like objects of matching shapes except for the first axis.
    :param seq_len: the value of T. It must be the maximum of the lengths Ti of the arrays at
    minimum. Will default to that value if None.
    :param pad_value: the value to pad the arrays with.
    :return: a (B, T, ...) tensor
    """
    if seq_len is None:
        seq_len = max(len(array) for array in arrays)
    else:
        assert seq_len >= max(len(array) for array in arrays)

    # Convert lists to np.array
    if isinstance(arrays[0], list):
        arrays = [np.array(array) for array in arrays]

    # Convert to tensor and handle device
    device = None
    if isinstance(arrays[0], torch.Tensor):
        tensors = arrays
        device = tensors[0].device
    else:
        tensors = [torch.as_tensor(array) for array in arrays]

    # Fill the packed tensor with the array data
    packed_shape = (len(tensors), seq_len, *tensors[0].shape[1:])
    packed_tensor = torch.full(packed_shape, pad_value, dtype=tensors[0].dtype, device=device)

    for i, tensor in enumerate(tensors):
        packed_tensor[i, :tensor.size(0)] = tensor

    return packed_tensor


def get_num_wins(
    n_frames: int,
    step: int,
    min_coverage: float,
    hp: VoiceEncConfig,
):
    assert n_frames > 0
    win_size = hp.ve_partial_frames
    n_wins, remainder = divmod(max(n_frames - win_size + step, 0), step)
    if n_wins == 0 or (remainder + (win_size - step)) / win_size >= min_coverage:
        n_wins += 1
    target_n = win_size + step * (n_wins - 1)
    return n_wins, target_n


def get_frame_step(
    overlap: float,
    rate: float,
    hp: VoiceEncConfig,
):
    # Compute how many frames separate two partial utterances
    assert 0 <= overlap < 1
    if rate is None:
        frame_step = int(np.round(hp.ve_partial_frames * (1 - overlap)))
    else:
        frame_step = int(np.round((hp.sample_rate / rate) / hp.ve_partial_frames))
    assert 0 < frame_step <= hp.ve_partial_frames
    return frame_step


def stride_as_partials(
    mel: np.ndarray,
    hp: VoiceEncConfig,
    overlap=0.5,
    rate: float=None,
    min_coverage=0.8,
):
    """
    Takes unscaled mels in (T, M) format
    TODO: doc
    """
    assert 0 < min_coverage <= 1
    frame_step = get_frame_step(overlap, rate, hp)

    # Compute how many partials can fit in the mel
    n_partials, target_len = get_num_wins(len(mel), frame_step, min_coverage, hp)

    # Trim or pad the mel spectrogram to match the number of partials
    if target_len > len(mel):
        mel = np.concatenate((mel, np.full((target_len - len(mel), hp.num_mels), 0)))
    elif target_len < len(mel):
        mel = mel[:target_len]

    # Ensure the numpy array data is float32 and contiguous in memory
    mel = mel.astype(np.float32, order="C")

    # Re-arrange the array in memory to be of shape (N, P, M) with partials overlapping eachother,
    # where N is the number of partials, P is the number of frames of each partial and M the
    # number of channels of the mel spectrograms.
    shape = (n_partials, hp.ve_partial_frames, hp.num_mels)
    strides = (mel.strides[0] * frame_step, mel.strides[0], mel.strides[1])
    partials = as_strided(mel, shape, strides)
    return partials


class VoiceEncoder(nn.Module):
    def __init__(self, hp=VoiceEncConfig()):
        super().__init__()

        self.hp = hp

        # Network definition
        self.lstm = nn.LSTM(self.hp.num_mels, self.hp.ve_hidden_size, num_layers=3, batch_first=True)
        if hp.flatten_lstm_params:
            self.lstm.flatten_parameters()
        self.proj = nn.Linear(self.hp.ve_hidden_size, self.hp.speaker_embed_size)

        # Cosine similarity scaling (fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.]), requires_grad=True)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.]), requires_grad=True)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, mels: torch.FloatTensor):
        """
        Computes the embeddings of a batch of partial utterances.

        :param mels: a batch of unscaled mel spectrograms of same duration as a float32 tensor
        of shape (B, T, M) where T is hp.ve_partial_frames
        :return: the embeddings as a float32 tensor of shape (B, E) where E is
        hp.speaker_embed_size. Embeddings are L2-normed and thus lay in the range [-1, 1].
        """
        if self.hp.normalized_mels and (mels.min() < 0 or mels.max() > 1):
            raise Exception(f"Mels outside [0, 1]. Min={mels.min()}, Max={mels.max()}")

        # Pass the input through the LSTM layers
        _, (hidden, _) = self.lstm(mels)

        # Project the final hidden state
        raw_embeds = self.proj(hidden[-1])
        if self.hp.ve_final_relu:
            raw_embeds = F.relu(raw_embeds)

        # L2 normalize the embeddings.
        return raw_embeds / torch.linalg.norm(raw_embeds, dim=1, keepdim=True)

    def inference(self, mels: torch.Tensor, mel_lens, overlap=0.5, rate: float=None, min_coverage=0.8, batch_size=None):
        """
        Computes the embeddings of a batch of full utterances with gradients.

        :param mels: (B, T, M) unscaled mels
        :return: (B, E) embeddings on CPU
        """
        mel_lens = mel_lens.tolist() if torch.is_tensor(mel_lens) else mel_lens

        # Compute where to split the utterances into partials
        frame_step = get_frame_step(overlap, rate, self.hp)
        n_partials, target_lens = zip(*(get_num_wins(l, frame_step, min_coverage, self.hp) for l in mel_lens))

        # Possibly pad the mels to reach the target lengths
        len_diff = max(target_lens) - mels.size(1)
        if len_diff > 0:
            pad = torch.full((mels.size(0), len_diff, self.hp.num_mels), 0, dtype=torch.float32)
            mels = torch.cat((mels, pad.to(mels.device)), dim=1)

        # Group all partials together so that we can batch them easily
        partials = [
            mel[i * frame_step: i * frame_step + self.hp.ve_partial_frames]
            for mel, n_partial in zip(mels, n_partials) for i in range(n_partial)
        ]
        assert all(partials[0].shape == partial.shape for partial in partials)
        partials = torch.stack(partials)

        # Forward the partials
        n_chunks = int(np.ceil(len(partials) / (batch_size or len(partials))))
        partial_embeds = torch.cat([self(batch) for batch in partials.chunk(n_chunks)], dim=0).cpu()

        # Reduce the partial embeds into full embeds and L2-normalize them
        slices = np.concatenate(([0], np.cumsum(n_partials)))
        raw_embeds = [torch.mean(partial_embeds[start:end], dim=0) for start, end in zip(slices[:-1], slices[1:])]
        raw_embeds = torch.stack(raw_embeds)
        embeds = raw_embeds / torch.linalg.norm(raw_embeds, dim=1, keepdim=True)

        return embeds

    @staticmethod
    def utt_to_spk_embed(utt_embeds: np.ndarray):
        """
        Takes an array of L2-normalized utterance embeddings, computes the mean embedding and L2-normalize it to get a
        speaker embedding.
        """
        assert utt_embeds.ndim == 2
        utt_embeds = np.mean(utt_embeds, axis=0)
        return utt_embeds / np.linalg.norm(utt_embeds, 2)

    @staticmethod
    def voice_similarity(embeds_x: np.ndarray, embeds_y: np.ndarray):
        """
        Cosine similarity for L2-normalized utterance embeddings or speaker embeddings
        """
        embeds_x = embeds_x if embeds_x.ndim == 1 else VoiceEncoder.utt_to_spk_embed(embeds_x)
        embeds_y = embeds_y if embeds_y.ndim == 1 else VoiceEncoder.utt_to_spk_embed(embeds_y)
        return embeds_x @ embeds_y

    def embeds_from_mels(
        self, mels: Union[Tensor, List[np.ndarray]], mel_lens=None, as_spk=False, batch_size=32, **kwargs
    ):
        """
        Convenience function for deriving utterance or speaker embeddings from mel spectrograms.

        :param mels: unscaled mels strictly within [0, 1] as either a (B, T, M) tensor or a list of (Ti, M) arrays.
        :param mel_lens: if passing mels as a tensor, individual mel lengths
        :param as_spk: whether to return utterance embeddings or a single speaker embedding
        :param kwargs: args for inference()

        :returns: embeds as a (B, E) float32 numpy array if <as_spk> is False, else as a (E,) array
        """
        # Load mels in memory and pack them
        if isinstance(mels, List):
            mels = [np.asarray(mel) for mel in mels]
            assert all(m.shape[1] == mels[0].shape[1] for m in mels), "Mels aren't in (B, T, M) format"
            mel_lens = [mel.shape[0] for mel in mels]
            mels = pack(mels)

        # Embed them
        with torch.inference_mode():
            utt_embeds = self.inference(mels.to(self.device), mel_lens, batch_size=batch_size, **kwargs).numpy()

        return self.utt_to_spk_embed(utt_embeds) if as_spk else utt_embeds

    def embeds_from_wavs(
        self,
        wavs: List[np.ndarray],
        sample_rate,
        as_spk=False,
        batch_size=32,
        trim_top_db: Optional[float]=20,
        **kwargs
    ):
        """
        Wrapper around embeds_from_mels

        :param trim_top_db: this argument was only added for the sake of compatibility with metavoice's implementation
        """
        if sample_rate != self.hp.sample_rate:
            wavs = [
                librosa.resample(wav, orig_sr=sample_rate, target_sr=self.hp.sample_rate, res_type="kaiser_fast")
                for wav in wavs
            ]

        if trim_top_db:
            wavs = [librosa.effects.trim(wav, top_db=trim_top_db)[0] for wav in wavs]

        if "rate" not in kwargs:
            kwargs["rate"] = 1.3  # Resemble's default value.

        mels = [melspectrogram(w, self.hp).T for w in wavs]

        return self.embeds_from_mels(mels, as_spk=as_spk, batch_size=batch_size, **kwargs)
