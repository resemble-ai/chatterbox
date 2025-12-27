# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
MLX implementation of Subsampling layers for Conformer.
Port of PyTorch implementation from s3gen/transformer/subsampling.py
"""

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


class BaseSubsamplingMLX(nn.Module):
    """Base class for subsampling layers."""

    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1
        self.pos_enc = None

    def position_encoding(self, offset: int, size: int) -> mx.array:
        """Get position encoding."""
        if self.pos_enc is not None:
            return self.pos_enc.position_encoding(offset, size)
        return mx.zeros((1, size, 1))


class LinearNoSubsamplingMLX(BaseSubsamplingMLX):
    """Linear transform input without subsampling for MLX.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc_class: Positional encoding class instance.
    """

    def __init__(
        self,
        idim: int,
        odim: int,
        dropout_rate: float = 0.0,
        pos_enc_class: nn.Module = None,
    ):
        """Construct a LinearNoSubsamplingMLX object."""
        super().__init__()
        self.linear = nn.Linear(idim, odim)
        self.norm = nn.LayerNorm(odim, eps=1e-5)
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def __call__(
        self,
        x: mx.array,
        x_mask: mx.array,
        offset: int = 0,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Input x.

        Args:
            x: Input tensor (batch, time, idim).
            x_mask: Input mask (batch, 1, time).
            offset: Position offset.

        Returns:
            Output tensor (batch, time, odim).
            Position embedding.
            Output mask (batch, 1, time).
        """
        x = self.linear(x)
        x = self.norm(x)

        if self.pos_enc is not None:
            x, pos_emb = self.pos_enc(x, offset)
        else:
            pos_emb = mx.zeros((1, x.shape[1], x.shape[2]))

        return x, pos_emb, x_mask


class EmbeddingNoSubsamplingMLX(BaseSubsamplingMLX):
    """Embedding input without subsampling for MLX.

    Args:
        idim (int): Vocabulary size.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc_class: Positional encoding class instance.
    """

    def __init__(
        self,
        idim: int,
        odim: int,
        dropout_rate: float = 0.0,
        pos_enc_class: nn.Module = None,
    ):
        """Construct an EmbeddingNoSubsamplingMLX object."""
        super().__init__()
        self.embed = nn.Embedding(idim, odim)
        self.pos_enc = pos_enc_class

    def __call__(
        self,
        x: mx.array,
        x_mask: mx.array,
        offset: int = 0,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Input x.

        Args:
            x: Input tensor (batch, time).
            x_mask: Input mask (batch, 1, time).
            offset: Position offset.

        Returns:
            Output tensor (batch, time, odim).
            Position embedding.
            Output mask.
        """
        x = self.embed(x)

        if self.pos_enc is not None:
            x, pos_emb = self.pos_enc(x, offset)
        else:
            pos_emb = mx.zeros((1, x.shape[1], x.shape[2]))

        return x, pos_emb, x_mask


class Conv1dSubsampling2MLX(BaseSubsamplingMLX):
    """Convolutional 1D subsampling (to 1/2 length) for MLX.

    Designed for Whisper-style subsampling.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc_class: Positional encoding class instance.
    """

    def __init__(
        self,
        idim: int,
        odim: int,
        dropout_rate: float = 0.0,
        pos_enc_class: nn.Module = None,
    ):
        """Construct a Conv1dSubsampling2MLX object."""
        super().__init__()

        # Note: MLX Conv1d expects (batch, seq, channels) by default
        # but we'll transpose to match PyTorch convention
        self.conv1 = nn.Conv1d(idim, odim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(odim, odim, kernel_size=3, stride=2, padding=1)

        self.pos_enc = pos_enc_class
        self.subsampling_rate = 2
        self.right_context = 4  # (3-1)*1 + (3-1)*1

    def __call__(
        self,
        x: mx.array,
        x_mask: mx.array,
        offset: int = 0,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Input x.

        Args:
            x: Input tensor (batch, time, idim).
            x_mask: Input mask (batch, 1, time).
            offset: Position offset.

        Returns:
            Output tensor (batch, time//2, odim).
            Position embedding.
            Output mask (batch, 1, time//2).
        """
        # Transpose for conv: (batch, time, channels) -> (batch, channels, time)
        x = mx.transpose(x, (0, 2, 1))

        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))

        # Transpose back: (batch, channels, time) -> (batch, time, channels)
        x = mx.transpose(x, (0, 2, 1))

        if self.pos_enc is not None:
            x, pos_emb = self.pos_enc(x, offset)
        else:
            pos_emb = mx.zeros((1, x.shape[1], x.shape[2]))

        # Update mask for subsampling
        x_mask = x_mask[:, :, ::2]

        return x, pos_emb, x_mask
