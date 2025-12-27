# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
MLX implementation of Conditional Decoder (U-Net style).
Port of PyTorch implementation from s3gen/decoder.py

The structure mirrors PyTorch's nn.ModuleList for proper weight loading:
- down_blocks: List of [resnet, transformer_blocks, downsample]
- mid_blocks: List of [resnet, transformer_blocks]
- up_blocks: List of [resnet, transformer_blocks, upsample]
"""

from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .matcha.decoder_mlx import (
    SinusoidalPosEmbMLX,
    TimestepEmbeddingMLX,
    ResnetBlock1DMLX,
    Block1DMLX,
    CausalBlock1DMLX,
    Downsample1DMLX,
    Upsample1DMatchaMLX,
    CausalConv1DMLX,
)
from .matcha.transformer_mlx import BasicTransformerBlockMLX
from .utils.mask_mlx import add_optional_chunk_mask, mask_to_bias


class DownBlockMLX(nn.Module):
    """A single down block containing resnet, transformer blocks, and downsample.

    Structure matches PyTorch's nn.ModuleList([resnet, transformer_blocks, downsample])
    so that weight names align: down_blocks.0.0 = resnet, down_blocks.0.1 = transformers, etc.
    """

    def __init__(
        self,
        resnet: ResnetBlock1DMLX,
        transformer_blocks: List[BasicTransformerBlockMLX],
        downsample: nn.Module,
    ):
        super().__init__()
        # Use indices to match PyTorch nn.ModuleList structure
        # down_blocks.X.0 = resnet
        # down_blocks.X.1 = transformer_blocks (ModuleList)
        # down_blocks.X.2 = downsample
        self.layers_0 = resnet  # Will be named as "0" in parameter tree
        self.layers_2 = downsample

        # Transformer blocks with explicit indexing
        self.num_transformer_blocks = len(transformer_blocks)
        for i, block in enumerate(transformer_blocks):
            setattr(self, f"layers_1_{i}", block)

    @property
    def resnet(self):
        return self.layers_0

    @property
    def transformer_blocks(self):
        return [
            getattr(self, f"layers_1_{i}") for i in range(self.num_transformer_blocks)
        ]

    @property
    def downsample(self):
        return self.layers_2


class MidBlockMLX(nn.Module):
    """A single mid block containing resnet and transformer blocks.

    Structure matches PyTorch's nn.ModuleList([resnet, transformer_blocks])
    """

    def __init__(
        self,
        resnet: ResnetBlock1DMLX,
        transformer_blocks: List[BasicTransformerBlockMLX],
    ):
        super().__init__()
        self.layers_0 = resnet

        # Transformer blocks with explicit indexing
        self.num_transformer_blocks = len(transformer_blocks)
        for i, block in enumerate(transformer_blocks):
            setattr(self, f"layers_1_{i}", block)

    @property
    def resnet(self):
        return self.layers_0

    @property
    def transformer_blocks(self):
        return [
            getattr(self, f"layers_1_{i}") for i in range(self.num_transformer_blocks)
        ]


class UpBlockMLX(nn.Module):
    """A single up block containing resnet, transformer blocks, and upsample.

    Structure matches PyTorch's nn.ModuleList([resnet, transformer_blocks, upsample])
    """

    def __init__(
        self,
        resnet: ResnetBlock1DMLX,
        transformer_blocks: List[BasicTransformerBlockMLX],
        upsample: nn.Module,
    ):
        super().__init__()
        self.layers_0 = resnet
        self.layers_2 = upsample

        # Transformer blocks with explicit indexing
        self.num_transformer_blocks = len(transformer_blocks)
        for i, block in enumerate(transformer_blocks):
            setattr(self, f"layers_1_{i}", block)

    @property
    def resnet(self):
        return self.layers_0

    @property
    def transformer_blocks(self):
        return [
            getattr(self, f"layers_1_{i}") for i in range(self.num_transformer_blocks)
        ]

    @property
    def upsample(self):
        return self.layers_2


class ConditionalDecoderMLX(nn.Module):
    """U-Net style conditional decoder for MLX.

    This decoder processes mel-spectrograms conditioned on:
    - Time step t for flow matching
    - Speaker embedding
    - Conditioning features from encoder
    """

    def __init__(
        self,
        in_channels: int = 320,
        out_channels: int = 80,
        causal: bool = True,
        channels: List[int] = [256],
        dropout: float = 0.0,
        attention_head_dim: int = 64,
        n_blocks: int = 4,
        num_mid_blocks: int = 12,
        num_heads: int = 8,
        act_fn: str = "gelu",
    ):
        """Initialize ConditionalDecoderMLX.

        Args:
            in_channels: Input channels (mel + condition).
            out_channels: Output mel channels.
            causal: Whether to use causal convolutions.
            channels: Channel sizes for each level.
            dropout: Dropout rate.
            attention_head_dim: Dimension per attention head.
            n_blocks: Number of transformer blocks per level.
            num_mid_blocks: Number of mid blocks.
            num_heads: Number of attention heads.
            act_fn: Activation function for transformer.
        """
        super().__init__()

        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = causal
        self.static_chunk_size = 0

        # Time embeddings
        self.time_embeddings = SinusoidalPosEmbMLX(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbeddingMLX(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )

        # Build U-Net blocks - use explicit attributes for MLX parameter tracking
        # Down blocks
        self.num_down_blocks = len(channels)
        output_channel = in_channels
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1

            # ResNet block
            resnet = ResnetBlock1DMLX(
                dim=input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
                causal=causal,
            )

            # Transformer blocks
            transformer_blocks = [
                BasicTransformerBlockMLX(
                    dim=output_channel,
                    num_attention_heads=num_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn=act_fn,
                )
                for _ in range(n_blocks)
            ]

            # Downsample
            if is_last:
                if causal:
                    downsample = CausalConv1DMLX(output_channel, output_channel, 3)
                else:
                    downsample = nn.Conv1d(
                        output_channel, output_channel, kernel_size=3, padding=1
                    )
            else:
                downsample = Downsample1DMLX(output_channel)

            # Use explicit attribute name for MLX tracking
            setattr(
                self,
                f"down_blocks_{i}",
                DownBlockMLX(resnet, transformer_blocks, downsample),
            )

        # Mid blocks
        self.num_mid_blocks = num_mid_blocks
        for i in range(num_mid_blocks):
            resnet = ResnetBlock1DMLX(
                dim=channels[-1],
                dim_out=channels[-1],
                time_emb_dim=time_embed_dim,
                causal=causal,
            )

            transformer_blocks = [
                BasicTransformerBlockMLX(
                    dim=channels[-1],
                    num_attention_heads=num_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn=act_fn,
                )
                for _ in range(n_blocks)
            ]

            setattr(self, f"mid_blocks_{i}", MidBlockMLX(resnet, transformer_blocks))

        # Up blocks
        reversed_channels = channels[::-1] + (channels[0],)
        self.num_up_blocks = len(reversed_channels) - 1
        for i in range(len(reversed_channels) - 1):
            input_channel = reversed_channels[i] * 2  # Skip connection doubles channels
            output_channel = reversed_channels[i + 1]
            is_last = i == len(reversed_channels) - 2

            resnet = ResnetBlock1DMLX(
                dim=input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
                causal=causal,
            )

            transformer_blocks = [
                BasicTransformerBlockMLX(
                    dim=output_channel,
                    num_attention_heads=num_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn=act_fn,
                )
                for _ in range(n_blocks)
            ]

            if is_last:
                if causal:
                    upsample = CausalConv1DMLX(output_channel, output_channel, 3)
                else:
                    upsample = nn.Conv1d(
                        output_channel, output_channel, kernel_size=3, padding=1
                    )
            else:
                upsample = Upsample1DMatchaMLX(output_channel, use_conv_transpose=True)

            setattr(
                self, f"up_blocks_{i}", UpBlockMLX(resnet, transformer_blocks, upsample)
            )

        # Final layers
        if causal:
            self.final_block = CausalBlock1DMLX(
                reversed_channels[-1], reversed_channels[-1]
            )
        else:
            self.final_block = Block1DMLX(reversed_channels[-1], reversed_channels[-1])

        self.final_proj = nn.Conv1d(reversed_channels[-1], out_channels, kernel_size=1)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        mu: mx.array,
        t: mx.array,
        spks: Optional[mx.array] = None,
        cond: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass of U-Net conditional decoder.

        Args:
            x: Noisy mel input (batch, mel_channels, time).
            mask: Output mask (batch, 1, time).
            mu: Encoder output (batch, mel_channels, time).
            t: Time step (batch,).
            spks: Speaker embedding (batch, spk_dim).
            cond: Additional conditioning (batch, cond_dim, time).

        Returns:
            Predicted velocity field (batch, mel_channels, time).
        """
        # Time embedding
        t_emb = self.time_embeddings(t).astype(t.dtype)
        t_emb = self.time_mlp(t_emb)

        # Pack inputs: [x, mu]
        x = mx.concatenate([x, mu], axis=1)

        # Add speaker embedding
        if spks is not None:
            # Expand to time dimension
            spks_expanded = mx.broadcast_to(
                mx.expand_dims(spks, axis=-1),
                (spks.shape[0], spks.shape[1], x.shape[-1]),
            )
            x = mx.concatenate([x, spks_expanded], axis=1)

        # Add conditioning
        if cond is not None:
            x = mx.concatenate([x, cond], axis=1)

        # Track hiddens and masks for skip connections
        hiddens = []
        masks = [mask]

        # Down blocks
        for i in range(self.num_down_blocks):
            block = getattr(self, f"down_blocks_{i}")
            mask_down = masks[-1]
            x = block.resnet(x, mask_down, t_emb)

            # Transformer blocks
            x = mx.transpose(x, (0, 2, 1))  # (B, T, C)
            attn_mask = add_optional_chunk_mask(
                x,
                mask_down.astype(mx.bool_),
                False,
                False,
                0,
                self.static_chunk_size,
                -1,
            )
            attn_mask = mask_to_bias(attn_mask, x.dtype)

            for transformer in block.transformer_blocks:
                x = transformer(x, attention_mask=attn_mask, timestep=t_emb)

            x = mx.transpose(x, (0, 2, 1))  # (B, C, T)

            hiddens.append(x)
            x = block.downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        # Mid blocks
        for i in range(self.num_mid_blocks):
            block = getattr(self, f"mid_blocks_{i}")
            x = block.resnet(x, mask_mid, t_emb)

            x = mx.transpose(x, (0, 2, 1))
            attn_mask = add_optional_chunk_mask(
                x,
                mask_mid.astype(mx.bool_),
                False,
                False,
                0,
                self.static_chunk_size,
                -1,
            )
            attn_mask = mask_to_bias(attn_mask, x.dtype)

            for transformer in block.transformer_blocks:
                x = transformer(x, attention_mask=attn_mask, timestep=t_emb)

            x = mx.transpose(x, (0, 2, 1))

        # Up blocks
        for i in range(self.num_up_blocks):
            block = getattr(self, f"up_blocks_{i}")
            mask_up = masks.pop()
            skip = hiddens.pop()

            # Concatenate skip connection (align lengths)
            x = mx.concatenate([x[:, :, : skip.shape[-1]], skip], axis=1)

            x = block.resnet(x, mask_up, t_emb)

            x = mx.transpose(x, (0, 2, 1))
            attn_mask = add_optional_chunk_mask(
                x, mask_up.astype(mx.bool_), False, False, 0, self.static_chunk_size, -1
            )
            attn_mask = mask_to_bias(attn_mask, x.dtype)

            for transformer in block.transformer_blocks:
                x = transformer(x, attention_mask=attn_mask, timestep=t_emb)

            x = mx.transpose(x, (0, 2, 1))
            x = block.upsample(x * mask_up)

        # Final block and projection
        x = self.final_block(x, mask_up)

        # final_proj is nn.Conv1d which expects MLX format (B, T, C)
        x = x * mask_up
        x = mx.transpose(x, (0, 2, 1))  # [B, C, T] -> [B, T, C]
        output = self.final_proj(x)
        output = mx.transpose(output, (0, 2, 1))  # [B, T, C] -> [B, C, T]

        return output * mask
