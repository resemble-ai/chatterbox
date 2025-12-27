# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
MLX implementation of Conformer Encoder Layer.
Port of PyTorch implementation from s3gen/transformer/encoder_layer.py
"""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class TransformerEncoderLayerMLX(nn.Module):
    """Standard Transformer encoder layer for MLX.

    Args:
        size (int): Input dimension.
        self_attn: Self-attention module.
        feed_forward: Feed-forward module.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use pre-norm.
    """

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        feed_forward: nn.Module,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
    ):
        """Construct a TransformerEncoderLayerMLX object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, eps=1e-12)
        self.norm2 = nn.LayerNorm(size, eps=1e-12)
        self.size = size
        self.normalize_before = normalize_before

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        pos_emb: Optional[mx.array] = None,
        mask_pad: Optional[mx.array] = None,
        att_cache: Optional[mx.array] = None,
        cnn_cache: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """Compute encoded features.

        Args:
            x: Input tensor (batch, time, size).
            mask: Mask tensor (batch, time, time).
            pos_emb: Positional embedding.
            mask_pad: Padding mask (batch, 1, time).
            att_cache: Attention cache.
            cnn_cache: CNN cache (not used in Transformer).

        Returns:
            Output tensor (batch, time, size).
            Mask tensor.
            Attention cache.
            CNN cache (zeros).
        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        x_att, new_att_cache = self.self_attn(
            x, x, x, mask=mask, pos_emb=pos_emb, cache=att_cache
        )
        x = residual + x_att

        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)

        x = residual + self.feed_forward(x)

        if not self.normalize_before:
            x = self.norm2(x)

        # Return empty CNN cache for interface compatibility
        fake_cnn_cache = mx.zeros((0, 0, 0))
        return x, mask, new_att_cache, fake_cnn_cache


class ConformerEncoderLayerMLX(nn.Module):
    """Conformer encoder layer for MLX.

    Structure: FFN (macaron) -> MHA -> Conv -> FFN -> LayerNorm

    Args:
        size (int): Input dimension.
        self_attn: Self-attention module.
        feed_forward: Feed-forward module.
        feed_forward_macaron: Macaron-style feed-forward module.
        conv_module: Convolution module.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use pre-norm.
    """

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        feed_forward: Optional[nn.Module] = None,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
    ):
        """Construct a ConformerEncoderLayerMLX object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module

        self.norm_ff = nn.LayerNorm(size, eps=1e-12)
        self.norm_mha = nn.LayerNorm(size, eps=1e-12)

        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-12)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0

        if conv_module is not None:
            self.norm_conv = nn.LayerNorm(size, eps=1e-12)
            self.norm_final = nn.LayerNorm(size, eps=1e-12)

        self.size = size
        self.normalize_before = normalize_before

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        pos_emb: Optional[mx.array] = None,
        mask_pad: Optional[mx.array] = None,
        att_cache: Optional[mx.array] = None,
        cnn_cache: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """Compute encoded features.

        Args:
            x: Input tensor (batch, time, size).
            mask: Attention mask (batch, time, time).
            pos_emb: Positional embedding.
            mask_pad: Padding mask (batch, 1, time).
            att_cache: Attention cache.
            cnn_cache: Convolution cache.

        Returns:
            Output tensor (batch, time, size).
            Mask tensor.
            New attention cache.
            New CNN cache.
        """
        # Macaron-style FFN (first half)
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.feed_forward_macaron(x)
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # Multi-headed self-attention
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(
            x, x, x, mask=mask, pos_emb=pos_emb, cache=att_cache
        )
        x = residual + x_att
        if not self.normalize_before:
            x = self.norm_mha(x)

        # Convolution module
        new_cnn_cache = mx.zeros((0, 0, 0))
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + x
            if not self.normalize_before:
                x = self.norm_conv(x)

        # Feed-forward module (second half or full)
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm_ff(x)

        # Final layer norm for Conformer
        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache
