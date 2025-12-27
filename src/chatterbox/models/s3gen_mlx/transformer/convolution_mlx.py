# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
MLX implementation of Convolution Module for Conformer encoder.
Port of PyTorch convolution from s3gen/transformer/convolution.py
"""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


def glu(x: mx.array, axis: int = -1) -> mx.array:
    """Gated Linear Unit activation.

    Splits input along axis and returns first_half * sigmoid(second_half).
    """
    a, b = mx.split(x, 2, axis=axis)
    return a * mx.sigmoid(b)


class ConvolutionModuleMLX(nn.Module):
    """Convolution Module in Conformer model for MLX.

    This module performs:
    1. Pointwise conv to expand channels (2x)
    2. GLU activation
    3. Depthwise separable convolution
    4. Normalization + activation
    5. Pointwise conv to restore channels
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        activation: str = "relu",
        norm: str = "layer_norm",
        causal: bool = False,
        bias: bool = True,
    ):
        """Construct a ConvolutionModuleMLX object.

        Args:
            channels: The number of channels of conv layers.
            kernel_size: Kernel size of conv layers.
            activation: Activation function name.
            norm: Normalization type ('batch_norm' or 'layer_norm').
            causal: Whether to use causal convolution.
            bias: Whether to use bias.
        """
        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size

        # Pointwise conv 1: expand to 2x channels for GLU
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        # Causal or symmetric padding
        if causal:
            self.padding = 0
            self.lorder = kernel_size - 1
        else:
            assert (
                kernel_size - 1
            ) % 2 == 0, "kernel_size should be odd for symmetric conv"
            self.padding = (kernel_size - 1) // 2
            self.lorder = 0

        # Depthwise conv
        self.depthwise_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding,
            # Note: MLX Conv1d doesn't support groups parameter directly
            # We'll implement depthwise as regular conv for now
            bias=bias,
        )

        # Normalization - use LayerNorm for MLX (BatchNorm has limited support)
        self.use_layer_norm = True  # Force LayerNorm for MLX
        self.norm = nn.LayerNorm(channels)

        # Pointwise conv 2: restore channels
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        # Activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish" or activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

    def __call__(
        self,
        x: mx.array,
        mask_pad: Optional[mx.array] = None,
        cache: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Compute convolution module.

        Args:
            x: Input tensor (batch, time, channels) - MLX channel-last format.
            mask_pad: Mask for batch padding (batch, 1, time).
            cache: Left context cache for causal conv (batch, cache_t, channels).

        Returns:
            Output tensor (batch, time, channels).
            New cache tensor.
        """
        # MLX uses channel-last format [B, T, C]
        # No transpose needed for Conv1d in MLX

        # Apply mask if provided (need to expand for channels)
        if mask_pad is not None and mask_pad.shape[-1] > 0:
            # mask_pad is [B, 1, T] - transpose and expand for [B, T, C]
            mask_expanded = mx.transpose(mask_pad, (0, 2, 1))  # [B, T, 1]
            x = mx.where(mask_expanded, x, mx.zeros_like(x))

        # Handle causal padding
        if self.lorder > 0:
            if cache is None or cache.shape[1] == 0:
                # Pad with zeros on the left
                x = mx.pad(x, [(0, 0), (self.lorder, 0), (0, 0)])
            else:
                # Concatenate cache (cache is [B, cache_t, C])
                x = mx.concatenate([cache, x], axis=1)

            # Save cache for next iteration [B, lorder, C]
            new_cache = x[:, -self.lorder :, :]
        else:
            new_cache = mx.zeros((x.shape[0], 0, x.shape[2]))

        # GLU mechanism
        # MLX Conv1d: [B, T, C_in] -> [B, T, C_out]
        x = self.pointwise_conv1(x)  # [B, T, 2*channel]
        x = glu(x, axis=-1)  # [B, T, channel]

        # Depthwise conv
        x = self.depthwise_conv(x)  # [B, T, channel]

        # LayerNorm operates on last dim, which is channels - correct for MLX
        x = self.activation(self.norm(x))

        # Final pointwise conv
        x = self.pointwise_conv2(x)  # [B, T, channel]

        # Apply mask if provided
        if mask_pad is not None and mask_pad.shape[-1] > 0:
            mask_expanded = mx.transpose(mask_pad, (0, 2, 1))  # [B, T, 1]
            x = mx.where(mask_expanded, x, mx.zeros_like(x))

        return x, new_cache
