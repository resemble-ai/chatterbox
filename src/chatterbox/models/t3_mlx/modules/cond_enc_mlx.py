# Copyright (c) 2025 MichaelYangAI
# MIT License

from dataclasses import dataclass
from typing import Optional
import mlx.core as mx
import mlx.nn as nn

from ..modules.perceiver_mlx import PerceiverMLX
from ...t3.modules.t3_config import T3Config


@dataclass
class T3CondMLX:
    """
    MLX version of T3Cond dataclass for conditioning information.
    Uses mx.array instead of torch.Tensor.
    """

    speaker_emb: mx.array
    clap_emb: Optional[mx.array] = None
    cond_prompt_speech_tokens: Optional[mx.array] = None
    cond_prompt_speech_emb: Optional[mx.array] = None
    emotion_adv: Optional[float] = 0.5

    def to_device(self):
        """
        MLX uses unified memory, so this is a no-op for compatibility.
        Returns self for API compatibility with PyTorch version.
        """
        return self

    @staticmethod
    def from_pytorch(t3_cond_pt):
        """
        Convert PyTorch T3Cond to MLX version.

        Args:
            t3_cond_pt: PyTorch T3Cond object

        Returns:
            T3CondMLX object
        """

        def to_mlx(tensor):
            if tensor is None:
                return None
            # Convert torch tensor to numpy, then to MLX
            return mx.array(tensor.cpu().numpy())

        # Handle emotion_adv - extract scalar value from tensor if needed
        emotion_adv = t3_cond_pt.emotion_adv
        if hasattr(emotion_adv, "item"):
            emotion_adv = float(emotion_adv.view(-1)[0].item())
        elif not isinstance(emotion_adv, (int, float)):
            emotion_adv = float(emotion_adv)

        return T3CondMLX(
            speaker_emb=to_mlx(t3_cond_pt.speaker_emb),
            clap_emb=to_mlx(t3_cond_pt.clap_emb),
            cond_prompt_speech_tokens=to_mlx(t3_cond_pt.cond_prompt_speech_tokens),
            cond_prompt_speech_emb=to_mlx(t3_cond_pt.cond_prompt_speech_emb),
            emotion_adv=emotion_adv,
        )


class T3CondEncMLX(nn.Module):
    """
    MLX implementation of T3 conditioning encoder.
    Handles all non-text conditioning: speaker embeddings, CLAP, emotion, etc.
    Converted from PyTorch version in cond_enc.py
    """

    def __init__(self, hp: T3Config):
        """
        Initialize conditioning encoder.

        Args:
            hp: T3Config object with hyperparameters
        """
        super().__init__()
        self.hp = hp

        # Speaker encoder
        if hp.encoder_type == "voice_encoder":
            self.spkr_enc = nn.Linear(hp.speaker_embed_size, hp.n_channels)
        else:
            raise NotImplementedError(f"Encoder type {hp.encoder_type} not implemented")

        # Emotion adversarial conditioning
        self.emotion_adv_fc = None
        if hp.emotion_adv:
            self.emotion_adv_fc = nn.Linear(1, hp.n_channels, bias=False)

        # Perceiver resampler
        self.perceiver = None
        if hp.use_perceiver_resampler:
            self.perceiver = PerceiverMLX()

    def __call__(self, cond: T3CondMLX) -> mx.array:
        """
        Forward pass for conditioning encoder.

        Args:
            cond: T3CondMLX object containing conditioning information

        Returns:
            Conditioning embeddings of shape (B, len_cond, dim)
        """
        # Validate inputs
        assert (cond.cond_prompt_speech_tokens is None) == (
            cond.cond_prompt_speech_emb is None
        ), "no embeddings for cond_prompt_speech_tokens"

        # Speaker embedding projection
        speaker_emb = mx.reshape(cond.speaker_emb, (-1, self.hp.speaker_embed_size))
        cond_spkr = mx.expand_dims(self.spkr_enc(speaker_emb), axis=1)  # (B, 1, dim)

        # Empty tensor for concatenation
        empty = mx.zeros((cond_spkr.shape[0], 0, cond_spkr.shape[2]))  # (B, 0, dim)

        # CLAP embedding (not implemented yet)
        assert cond.clap_emb is None, "clap_embed not implemented"
        cond_clap = empty  # (B, 0, dim)

        # Conditioning prompt speech embeddings
        cond_prompt_speech_emb = cond.cond_prompt_speech_emb
        if cond_prompt_speech_emb is None:
            cond_prompt_speech_emb = empty  # (B, 0, dim)
        elif self.hp.use_perceiver_resampler:
            cond_prompt_speech_emb = self.perceiver(cond_prompt_speech_emb)

        # Emotion adversarial conditioning
        cond_emotion_adv = empty  # (B, 0, dim)
        if self.hp.emotion_adv:
            assert cond.emotion_adv is not None, "emotion_adv must be provided"

            # Handle scalar or tensor emotion_adv
            if isinstance(cond.emotion_adv, (int, float)):
                # Get batch size from speaker embeddings and broadcast scalar
                batch_size = cond_spkr.shape[0]
                emotion_val = mx.full((batch_size, 1, 1), cond.emotion_adv)
            else:
                emotion_val = cond.emotion_adv
                # Reshape to (B, 1, 1)
                emotion_val = mx.reshape(emotion_val, (-1, 1, 1))

            cond_emotion_adv = self.emotion_adv_fc(emotion_val)

        # Concatenate all conditioning embeddings
        cond_embeds = mx.concatenate(
            [
                cond_spkr,
                cond_clap,
                cond_prompt_speech_emb,
                cond_emotion_adv,
            ],
            axis=1,
        )

        return cond_embeds
