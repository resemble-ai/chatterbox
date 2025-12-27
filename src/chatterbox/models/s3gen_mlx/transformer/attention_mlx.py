# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
MLX implementation of Multi-Head Attention layers for Conformer encoder.
Port of PyTorch attention from s3gen/transformer/attention.py
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class MultiHeadedAttentionMLX(nn.Module):
    """Multi-Head Attention layer for MLX.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate (not used in inference).
        key_bias (bool): Whether to use bias in key projection.
    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float = 0.0,
        key_bias: bool = True,
    ):
        """Construct a MultiHeadedAttentionMLX object."""
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        # Note: dropout not used in inference mode

    def _reshape_to_heads(self, x: mx.array) -> mx.array:
        """Reshape tensor for multi-head attention.

        Args:
            x: (batch, time, n_feat)

        Returns:
            (batch, n_head, time, d_k)
        """
        batch_size, time, _ = x.shape
        x = mx.reshape(x, (batch_size, time, self.h, self.d_k))
        return mx.transpose(x, (0, 2, 1, 3))  # (B, H, T, D)

    def _reshape_from_heads(self, x: mx.array) -> mx.array:
        """Reshape tensor back from multi-head attention.

        Args:
            x: (batch, n_head, time, d_k)

        Returns:
            (batch, time, n_feat)
        """
        batch_size, _, time, _ = x.shape
        x = mx.transpose(x, (0, 2, 1, 3))  # (B, T, H, D)
        return mx.reshape(x, (batch_size, time, self.h * self.d_k))

    def forward_qkv(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Transform query, key and value.

        Args:
            query: Query tensor (batch, time1, size).
            key: Key tensor (batch, time2, size).
            value: Value tensor (batch, time2, size).

        Returns:
            q: (batch, n_head, time1, d_k)
            k: (batch, n_head, time2, d_k)
            v: (batch, n_head, time2, d_k)
        """
        q = self._reshape_to_heads(self.linear_q(query))
        k = self._reshape_to_heads(self.linear_k(key))
        v = self._reshape_to_heads(self.linear_v(value))
        return q, k, v

    def forward_attention(
        self,
        value: mx.array,
        scores: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Compute attention context vector.

        Args:
            value: Transformed value (batch, n_head, time2, d_k).
            scores: Attention score (batch, n_head, time1, time2).
            mask: Mask (batch, 1, time2) or (batch, time1, time2).

        Returns:
            Transformed value (batch, time1, n_feat).
        """
        if mask is not None:
            # mask: True means valid, False means masked
            # We need to set masked positions to -inf
            if mask.shape[-1] > 0:
                mask = mx.expand_dims(mask, axis=1)  # (batch, 1, *, time2)
                # Adjust mask size if needed
                if mask.shape[-1] > scores.shape[-1]:
                    mask = mask[..., : scores.shape[-1]]
                # Convert boolean mask to attention bias
                # True (valid) -> 0, False (masked) -> -inf
                scores = mx.where(mask, scores, mx.array(float("-inf")))

        # Compute softmax in FP32 for numerical stability
        attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)

        # Apply attention to values
        x = attn @ value  # (batch, head, time1, d_k)

        # Reshape and project
        x = self._reshape_from_heads(x)
        return self.linear_out(x)

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: Optional[mx.array] = None,
        pos_emb: Optional[mx.array] = None,
        cache: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Compute scaled dot product attention.

        Args:
            query: Query tensor (batch, time1, size).
            key: Key tensor (batch, time2, size).
            value: Value tensor (batch, time2, size).
            mask: Mask tensor (batch, 1, time2) or (batch, time1, time2).
            pos_emb: Not used in standard attention.
            cache: Cache tensor (1, head, cache_t, d_k * 2).

        Returns:
            Output tensor (batch, time1, n_feat).
            Cache tensor (1, head, cache_t + time1, d_k * 2).
        """
        q, k, v = self.forward_qkv(query, key, value)

        # Handle KV cache
        if cache is not None and cache.shape[0] > 0:
            # Split cache into key and value parts
            cache_size = cache.shape[-1] // 2
            key_cache = cache[..., :cache_size]
            value_cache = cache[..., cache_size:]
            k = mx.concatenate([key_cache, k], axis=2)
            v = mx.concatenate([value_cache, v], axis=2)

        # New cache
        new_cache = mx.concatenate([k, v], axis=-1)

        # Compute attention scores
        scale = 1.0 / math.sqrt(self.d_k)
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) * scale

        output = self.forward_attention(v, scores, mask)
        return output, new_cache


class RelPositionMultiHeadedAttentionMLX(MultiHeadedAttentionMLX):
    """Multi-Head Attention with relative position encoding for MLX.

    Paper: https://arxiv.org/abs/1901.02860
    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float = 0.0,
        key_bias: bool = True,
    ):
        """Construct a RelPositionMultiHeadedAttentionMLX object."""
        super().__init__(n_head, n_feat, dropout_rate, key_bias)
        # Linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # Learnable bias for position attention
        self.pos_bias_u = mx.zeros((self.h, self.d_k))
        self.pos_bias_v = mx.zeros((self.h, self.d_k))

    def rel_shift(self, x: mx.array, zero_triu: bool = False) -> mx.array:
        """Compute relative positional encoding.

        This operation transforms position-based attention scores to handle
        relative position offsets correctly.

        Args:
            x: Input tensor (batch, head, time1, time_pos).
            zero_triu: Whether to zero out the upper triangular part.

        Returns:
            Output tensor (batch, head, time1, time1).
        """
        batch_size, n_head, time1, time_pos = x.shape

        # Pad with zeros at the beginning
        zero_pad = mx.zeros((batch_size, n_head, time1, 1))
        x_padded = mx.concatenate([zero_pad, x], axis=-1)  # [B, H, T1, time_pos+1]

        # Reshape to shift positions
        x_padded = mx.reshape(
            x_padded, (batch_size, n_head, -1, time1)
        )  # [B, H, time_pos+1, T1]

        # Slice to get the correct relative positions
        # We want time1 positions for each query position
        x = x_padded[:, :, 1 : time1 + 1, :]  # [B, H, T1, T1]
        x = mx.transpose(x, (0, 1, 3, 2))  # [B, H, T1, T1]

        if zero_triu:
            # Create upper triangular mask and apply
            ones = mx.ones((time1, time1))
            mask = mx.triu(ones, k=1)
            x = x * (1 - mask)

        return x

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: Optional[mx.array] = None,
        pos_emb: Optional[mx.array] = None,
        cache: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Compute scaled dot product attention with relative position encoding.

        Args:
            query: Query tensor (batch, time1, size).
            key: Key tensor (batch, time2, size).
            value: Value tensor (batch, time2, size).
            mask: Mask tensor.
            pos_emb: Positional embedding tensor (batch, time2, size).
            cache: Cache tensor.

        Returns:
            Output tensor (batch, time1, n_feat).
            Cache tensor.
        """
        q, k, v = self.forward_qkv(query, key, value)

        # Transpose q for position attention: (batch, time1, head, d_k)
        q_transposed = mx.transpose(q, (0, 2, 1, 3))

        # Handle KV cache
        if cache is not None and cache.shape[0] > 0:
            cache_size = cache.shape[-1] // 2
            key_cache = cache[..., :cache_size]
            value_cache = cache[..., cache_size:]
            k = mx.concatenate([key_cache, k], axis=2)
            v = mx.concatenate([value_cache, v], axis=2)

        new_cache = mx.concatenate([k, v], axis=-1)

        # Position encoding projection
        if pos_emb is not None:
            n_batch_pos = pos_emb.shape[0]
            p = self.linear_pos(pos_emb)
            p = mx.reshape(p, (n_batch_pos, -1, self.h, self.d_k))
            p = mx.transpose(p, (0, 2, 1, 3))  # (batch, head, time, d_k)

            # Add position biases
            q_with_bias_u = mx.transpose(q_transposed + self.pos_bias_u, (0, 2, 1, 3))
            q_with_bias_v = mx.transpose(q_transposed + self.pos_bias_v, (0, 2, 1, 3))

            # Compute matrix a and c (content-based attention)
            matrix_ac = q_with_bias_u @ mx.transpose(k, (0, 1, 3, 2))

            # Compute matrix b and d (position-based attention)
            matrix_bd = q_with_bias_v @ mx.transpose(p, (0, 1, 3, 2))

            # Always apply rel_shift to convert position attention scores
            # from [B, H, T1, 2*T1-1] to [B, H, T1, T1] format
            matrix_bd = self.rel_shift(matrix_bd)

            # Combine scores
            scale = 1.0 / math.sqrt(self.d_k)
            scores = (matrix_ac + matrix_bd) * scale
        else:
            # Fall back to standard attention without position encoding
            scale = 1.0 / math.sqrt(self.d_k)
            scores = (q @ mx.transpose(k, (0, 1, 3, 2))) * scale

        output = self.forward_attention(v, scores, mask)
        return output, new_cache
