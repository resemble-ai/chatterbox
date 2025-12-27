# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
MLX implementation of Matcha decoder components.
Port of PyTorch implementation from s3gen/matcha/decoder.py
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class SinusoidalPosEmbMLX(nn.Module):
    """Sinusoidal positional embeddings for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        assert dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def __call__(self, x: mx.array, scale: float = 1000.0) -> mx.array:
        """Generate sinusoidal embeddings.

        Args:
            x: Timestep tensor (batch,).
            scale: Scaling factor.

        Returns:
            Embeddings (batch, dim).
        """
        if x.ndim < 1:
            x = mx.expand_dims(x, axis=0)

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = mx.exp(mx.arange(half_dim, dtype=mx.float32) * -emb)
        emb = scale * mx.expand_dims(x, axis=1) * mx.expand_dims(emb, axis=0)
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
        return emb


class Block1DMLX(nn.Module):
    """1D convolution block with normalization and activation.

    Note: Interface uses PyTorch format (batch, channels, time) for compatibility,
    but internally converts to MLX format (batch, time, channels).
    """

    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim_out, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input (batch, channels, time) - PyTorch format.
            mask: Mask (batch, 1, time).

        Returns:
            Output (batch, dim_out, time) - PyTorch format.
        """
        # Convert to MLX format for Conv1d: [B, C, T] -> [B, T, C]
        x = mx.transpose(x, (0, 2, 1))
        mask_t = mx.transpose(mask, (0, 2, 1))  # [B, T, 1]

        x = self.conv(x * mask_t)

        # GroupNorm expects [B, T, C] which is what we have
        x = self.norm(x)
        x = nn.mish(x)

        # Apply mask and convert back to PyTorch format: [B, T, C] -> [B, C, T]
        x = x * mask_t
        x = mx.transpose(x, (0, 2, 1))
        return x


class CausalBlock1DMLX(nn.Module):
    """Causal 1D convolution block.

    Note: Interface uses PyTorch format (batch, channels, time) for compatibility,
    but internally converts to MLX format (batch, time, channels).
    """

    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim_out, kernel_size=3, padding=0)
        self.norm = nn.LayerNorm(dim_out)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        """Forward pass with causal padding.

        Args:
            x: Input (batch, channels, time) - PyTorch format.
            mask: Mask (batch, 1, time).

        Returns:
            Output (batch, dim_out, time) - PyTorch format.
        """
        # Convert to MLX format: [B, C, T] -> [B, T, C]
        x = mx.transpose(x, (0, 2, 1))
        mask_t = mx.transpose(mask, (0, 2, 1))  # [B, T, 1]

        x = x * mask_t
        # Causal padding: pad 2 on left (time dimension, which is now axis 1)
        x = mx.pad(x, [(0, 0), (2, 0), (0, 0)])
        x = self.conv(x)
        # LayerNorm on last dimension (channels) - already correct for MLX format
        x = self.norm(x)
        x = nn.mish(x)

        # Apply mask and convert back to PyTorch format: [B, T, C] -> [B, C, T]
        x = x * mask_t
        x = mx.transpose(x, (0, 2, 1))
        return x


class TimestepEmbeddingMLX(nn.Module):
    """Timestep embedding MLP."""

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)

        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

        if act_fn == "silu" or act_fn == "swish":
            self.act = nn.SiLU()
        elif act_fn == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.SiLU()

    def __call__(self, sample: mx.array) -> mx.array:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class ResnetBlock1DMLX(nn.Module):
    """ResNet block for 1D convolution.

    Note: Interface uses PyTorch format (batch, channels, time) for compatibility,
    but internally converts to MLX format (batch, time, channels).
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        time_emb_dim: int,
        groups: int = 8,
        causal: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out

        # MLP with explicit layers instead of Sequential
        # PyTorch: mlp.0 = SiLU (no params), mlp.1 = Linear
        self.mlp_0 = nn.SiLU()
        self.mlp_1 = nn.Linear(time_emb_dim, dim_out)

        if causal:
            self.block1 = CausalBlock1DMLX(dim, dim_out)
            self.block2 = CausalBlock1DMLX(dim_out, dim_out)
        else:
            self.block1 = Block1DMLX(dim, dim_out, groups)
            self.block2 = Block1DMLX(dim_out, dim_out, groups)

        self.res_conv = nn.Conv1d(dim, dim_out, kernel_size=1)

    def __call__(self, x: mx.array, mask: mx.array, time_emb: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input (batch, channels, time) - PyTorch format.
            mask: Mask (batch, 1, time).
            time_emb: Time embedding (batch, time_emb_dim).

        Returns:
            Output (batch, dim_out, time) - PyTorch format.
        """
        h = self.block1(x, mask)

        # Add time embedding: mlp output is [B, dim_out], expand to [B, dim_out, 1] for broadcast
        mlp_out = self.mlp_1(self.mlp_0(time_emb))
        h = h + mx.expand_dims(mlp_out, axis=-1)

        h = self.block2(h, mask)

        # Residual connection - res_conv needs format conversion
        # x is [B, C, T], convert to [B, T, C] for MLX Conv1d
        mask_t = mx.transpose(mask, (0, 2, 1))  # [B, T, 1]
        x_t = mx.transpose(x, (0, 2, 1))  # [B, T, C]
        res = self.res_conv(x_t * mask_t)  # [B, T, dim_out]
        res = mx.transpose(res, (0, 2, 1))  # [B, dim_out, T]

        output = h + res
        return output


class Downsample1DMLX(nn.Module):
    """1D downsampling layer.

    Note: Interface uses PyTorch format (batch, channels, time) for compatibility,
    but internally converts to MLX format (batch, time, channels).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input (batch, channels, time) - PyTorch format.

        Returns:
            Output (batch, channels, time//2) - PyTorch format.
        """
        # Convert to MLX format: [B, C, T] -> [B, T, C]
        x = mx.transpose(x, (0, 2, 1))
        x = self.conv(x)
        # Convert back to PyTorch format: [B, T, C] -> [B, C, T]
        x = mx.transpose(x, (0, 2, 1))
        return x


class Upsample1DMatchaMLX(nn.Module):
    """1D upsampling layer using transposed convolution.

    Note: Interface uses PyTorch format (batch, channels, time) for compatibility,
    but internally converts to MLX format (batch, time, channels).
    """

    def __init__(self, channels: int, use_conv_transpose: bool = True):
        super().__init__()
        self.channels = channels
        self.use_conv_transpose = use_conv_transpose

        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(
                channels, channels, kernel_size=4, stride=2, padding=1
            )
        else:
            self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def __call__(self, inputs: mx.array) -> mx.array:
        """Forward pass.

        Args:
            inputs: Input (batch, channels, time) - PyTorch format.

        Returns:
            Output (batch, channels, time*2) - PyTorch format.
        """
        # Convert to MLX format: [B, C, T] -> [B, T, C]
        x = mx.transpose(inputs, (0, 2, 1))

        if self.use_conv_transpose:
            x = self.conv(x)
        else:
            # Nearest neighbor upsampling
            x = mx.repeat(x, 2, axis=1)  # axis 1 is time in MLX format
            x = self.conv(x)

        # Convert back to PyTorch format: [B, T, C] -> [B, C, T]
        x = mx.transpose(x, (0, 2, 1))
        return x


class CausalConv1DMLX(nn.Module):
    """Causal 1D convolution.

    Note: Interface uses PyTorch format (batch, channels, time) for compatibility,
    but internally converts to MLX format (batch, time, channels).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        assert stride == 1, "Only stride=1 supported for causal conv"
        self.causal_padding = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=1, padding=0, bias=bias
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with causal padding.

        Args:
            x: Input (batch, channels, time) - PyTorch format.

        Returns:
            Output (batch, out_channels, time) - PyTorch format.
        """
        # Convert to MLX format: [B, C, T] -> [B, T, C]
        x = mx.transpose(x, (0, 2, 1))
        # Causal padding on time dimension (now axis 1)
        x = mx.pad(x, [(0, 0), (self.causal_padding, 0), (0, 0)])
        x = self.conv(x)
        # Convert back to PyTorch format: [B, T, C] -> [B, C, T]
        x = mx.transpose(x, (0, 2, 1))
        return x


class DecoderMLX(nn.Module):
    """
    U-Net style decoder for Matcha-TTS / CFM.

    This is a simplified version that works for inference.
    Uses ResNet blocks with time conditioning.

    Note: Interface uses PyTorch format (batch, channels, time) for compatibility.
    """

    def __init__(
        self,
        in_channels: int = 80,
        out_channels: int = 80,
        channels: tuple = (256, 256),
        dropout: float = 0.05,
        n_blocks: int = 1,
        num_mid_blocks: int = 2,
    ):
        super().__init__()

        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Time embedding
        self.time_embeddings = SinusoidalPosEmbMLX(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbeddingMLX(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )

        # Input projection: 2*in_channels (x + mu) -> in_channels
        self.input_proj = nn.Conv1d(2 * in_channels, in_channels, 1)

        # Down blocks
        self.down_blocks = []
        output_channel = in_channels
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1

            resnet = ResnetBlock1DMLX(
                dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim
            )
            if not is_last:
                downsample = Downsample1DMLX(output_channel)
            else:
                downsample = nn.Conv1d(output_channel, output_channel, 3, padding=1)

            self.down_blocks.append((resnet, downsample))

        # Mid blocks
        self.mid_blocks = []
        for i in range(num_mid_blocks):
            resnet = ResnetBlock1DMLX(
                dim=channels[-1], dim_out=channels[-1], time_emb_dim=time_embed_dim
            )
            self.mid_blocks.append(resnet)

        # Up blocks
        self.up_blocks = []
        channels_rev = channels[::-1] + (channels[0],)
        for i in range(len(channels_rev) - 1):
            input_channel = channels_rev[i]
            output_channel = channels_rev[i + 1]
            is_last = i == len(channels_rev) - 2

            resnet = ResnetBlock1DMLX(
                dim=2 * input_channel,  # Skip connection doubles channels
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
            )
            if not is_last:
                upsample = Upsample1DMatchaMLX(output_channel, use_conv_transpose=True)
            else:
                upsample = nn.Conv1d(output_channel, output_channel, 3, padding=1)

            self.up_blocks.append((resnet, upsample))

        # Final projection
        self.final_block = Block1DMLX(channels_rev[-1], channels_rev[-1])
        self.final_proj = nn.Conv1d(channels_rev[-1], self.out_channels, 1)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        t: mx.array,
        mu: mx.array,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            x: Noisy input [B, C, T] (PyTorch format).
            mask: Mask [B, 1, T].
            t: Time step [B].
            mu: Conditioning [B, C, T] (same as x).

        Returns:
            Predicted velocity [B, C, T].
        """
        # Time embedding
        t_emb = self.time_mlp(self.time_embeddings(t))

        # Concatenate x and mu
        x = mx.concatenate([x, mu], axis=1)  # [B, 2*C, T]

        # Input projection: [B, 2*C, T] -> [B, C, T]
        x_t = mx.transpose(x, (0, 2, 1))  # [B, T, 2*C]
        x_t = self.input_proj(x_t)  # [B, T, C]
        x = mx.transpose(x_t, (0, 2, 1))  # [B, C, T]

        # Store hidden states for skip connections
        hiddens = []

        # Down path
        for resnet, downsample in self.down_blocks:
            x = resnet(x, mask, t_emb)
            hiddens.append(x)
            # Handle downsample - might need format conversion
            x_t = mx.transpose(x, (0, 2, 1))  # [B, T, C] for MLX Conv1d
            x_t = (
                downsample(x_t)
                if isinstance(downsample, Downsample1DMLX)
                else downsample(x_t)
            )
            x = mx.transpose(x_t, (0, 2, 1))  # Back to [B, C, T]

        # Mid blocks
        for resnet in self.mid_blocks:
            x = resnet(x, mask, t_emb)

        # Up path
        for resnet, upsample in self.up_blocks:
            skip = hiddens.pop()
            x = mx.concatenate([x, skip], axis=1)
            x = resnet(x, mask, t_emb)
            # Handle upsample
            x_t = mx.transpose(x, (0, 2, 1))
            x_t = (
                upsample(x_t)
                if isinstance(upsample, Upsample1DMatchaMLX)
                else upsample(x_t)
            )
            x = mx.transpose(x_t, (0, 2, 1))

        # Final projection
        x = self.final_block(x, mask)
        x_t = mx.transpose(x, (0, 2, 1))
        x_t = self.final_proj(x_t)
        x = mx.transpose(x_t, (0, 2, 1))

        return x * mask
