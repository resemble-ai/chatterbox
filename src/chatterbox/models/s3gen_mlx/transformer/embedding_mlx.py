# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
MLX implementation of Positional Encoding for Conformer.
Port of PyTorch implementation from s3gen/transformer/embedding.py
"""

import math
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


class PositionalEncodingMLX(nn.Module):
    """Sinusoidal positional encoding for MLX.

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate (not used in inference).
        max_len (int): Maximum input length.
    """

    def __init__(
        self,
        d_model: int,
        dropout_rate: float = 0.0,
        max_len: int = 5000,
    ):
        """Construct a PositionalEncodingMLX object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(d_model)
        self.max_len = max_len

        # Pre-compute positional encodings
        pe = mx.zeros((max_len, d_model))
        position = mx.expand_dims(mx.arange(0, max_len, dtype=mx.float32), axis=1)
        div_term = mx.exp(
            mx.arange(0, d_model, 2, dtype=mx.float32) * -(math.log(10000.0) / d_model)
        )

        # Compute sin and cos embeddings
        sin_vals = mx.sin(position * div_term)
        cos_vals = mx.cos(position * div_term)

        # Interleave sin and cos: pe[:, 0::2] = sin, pe[:, 1::2] = cos
        # MLX doesn't support slice assignment, so we need to construct it differently
        pe_even = sin_vals  # (max_len, d_model//2)
        pe_odd = cos_vals  # (max_len, d_model//2)

        # Stack and reshape to interleave
        pe = mx.reshape(mx.stack([pe_even, pe_odd], axis=-1), (max_len, d_model))

        self.pe = mx.expand_dims(pe, axis=0)  # (1, max_len, d_model)

    def __call__(
        self,
        x: mx.array,
        offset: int = 0,
    ) -> Tuple[mx.array, mx.array]:
        """Add positional encoding.

        Args:
            x: Input tensor (batch, time, dim).
            offset: Position offset.

        Returns:
            Encoded tensor (batch, time, dim).
            Position embedding tensor.
        """
        pos_emb = self.position_encoding(offset, x.shape[1])
        x = x * self.xscale + pos_emb
        return x, pos_emb

    def position_encoding(self, offset: int, size: int) -> mx.array:
        """Get position encoding for given offset and size.

        Args:
            offset: Start offset.
            size: Required size.

        Returns:
            Position encoding tensor.
        """
        return self.pe[:, offset : offset + size]


class RelPositionalEncodingMLX(PositionalEncodingMLX):
    """Relative positional encoding for MLX.

    See: Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(
        self,
        d_model: int,
        dropout_rate: float = 0.0,
        max_len: int = 5000,
    ):
        """Construct a RelPositionalEncodingMLX object."""
        super().__init__(d_model, dropout_rate, max_len)
        # Override: don't add pos_emb to x for relative encoding

    def __call__(
        self,
        x: mx.array,
        offset: int = 0,
    ) -> Tuple[mx.array, mx.array]:
        """Compute positional encoding.

        Args:
            x: Input tensor (batch, time, dim).
            offset: Position offset.

        Returns:
            Scaled tensor (batch, time, dim).
            Position embedding tensor.
        """
        pos_emb = self.position_encoding(offset, x.shape[1])
        x = x * self.xscale
        return x, pos_emb


class EspnetRelPositionalEncodingMLX(nn.Module):
    """Relative positional encoding module (ESPnet style) for MLX.

    This uses both positive and negative position indices as described
    in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(
        self,
        d_model: int,
        dropout_rate: float = 0.0,
        max_len: int = 5000,
    ):
        """Construct an EspnetRelPositionalEncodingMLX object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(d_model)
        self.max_len = max_len
        self.pe = None
        self._init_pe(max_len)

    def _init_pe(self, max_len: int):
        """Initialize positional encodings."""
        # Positive positions
        pe_positive = mx.zeros((max_len, self.d_model))
        # Negative positions
        pe_negative = mx.zeros((max_len, self.d_model))

        position = mx.expand_dims(mx.arange(0, max_len, dtype=mx.float32), axis=1)
        div_term = mx.exp(
            mx.arange(0, self.d_model, 2, dtype=mx.float32)
            * -(math.log(10000.0) / self.d_model)
        )

        # Positive encodings
        sin_pos = mx.sin(position * div_term)
        cos_pos = mx.cos(position * div_term)
        pe_positive = mx.reshape(
            mx.stack([sin_pos, cos_pos], axis=-1), (max_len, self.d_model)
        )

        # Negative encodings
        sin_neg = mx.sin(-1 * position * div_term)
        cos_neg = mx.cos(-1 * position * div_term)
        pe_negative = mx.reshape(
            mx.stack([sin_neg, cos_neg], axis=-1), (max_len, self.d_model)
        )

        # Flip positive and concatenate
        # pe_positive is reversed, pe_negative starts from index 1
        pe_positive_flipped = pe_positive[::-1]  # Reverse
        pe_negative_trimmed = pe_negative[1:]  # Skip first element

        # Concatenate: [positive_reversed, negative_without_first]
        pe = mx.concatenate(
            [
                mx.expand_dims(pe_positive_flipped, axis=0),
                mx.expand_dims(pe_negative_trimmed, axis=0),
            ],
            axis=1,
        )

        self.pe = pe

    def __call__(
        self,
        x: mx.array,
        offset: int = 0,
    ) -> Tuple[mx.array, mx.array]:
        """Add positional encoding.

        Args:
            x: Input tensor (batch, time, dim).
            offset: Position offset.

        Returns:
            Scaled tensor (batch, time, dim).
            Position embedding tensor.
        """
        x = x * self.xscale
        pos_emb = self.position_encoding(size=x.shape[1], offset=offset)
        return x, pos_emb

    def position_encoding(self, size: int, offset: int = 0) -> mx.array:
        """Get position encoding.

        Args:
            size: Required size.
            offset: Start offset.

        Returns:
            Position encoding tensor.
        """
        center = self.pe.shape[1] // 2
        pos_emb = self.pe[:, center - size + 1 : center + size, :]
        return pos_emb


class NoPositionalEncodingMLX(nn.Module):
    """No position encoding - returns zeros."""

    def __init__(self, d_model: int, dropout_rate: float = 0.0):
        super().__init__()
        self.d_model = d_model

    def __call__(
        self,
        x: mx.array,
        offset: int = 0,
    ) -> Tuple[mx.array, mx.array]:
        """Return input and zero position embedding."""
        pos_emb = mx.zeros((1, x.shape[1], self.d_model))
        return x, pos_emb

    def position_encoding(self, offset: int, size: int) -> mx.array:
        return mx.zeros((1, size, self.d_model))
