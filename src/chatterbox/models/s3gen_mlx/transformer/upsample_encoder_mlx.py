# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
MLX implementation of Upsample Conformer Encoder.
Port of PyTorch implementation from s3gen/transformer/upsample_encoder.py
"""

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from .attention_mlx import MultiHeadedAttentionMLX, RelPositionMultiHeadedAttentionMLX
from .convolution_mlx import ConvolutionModuleMLX
from .encoder_layer_mlx import ConformerEncoderLayerMLX
from .feed_forward_mlx import PositionwiseFeedForwardMLX
from .embedding_mlx import EspnetRelPositionalEncodingMLX, RelPositionalEncodingMLX
from .subsampling_mlx import LinearNoSubsamplingMLX
from ..utils.mask_mlx import make_pad_mask, add_optional_chunk_mask


class Upsample1DMLX(nn.Module):
    """1D upsampling layer with convolution for MLX.

    Args:
        channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Upsampling factor.
    """

    def __init__(self, channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.stride = stride
        # Conv after interpolation
        self.conv = nn.Conv1d(
            channels, out_channels, kernel_size=stride * 2 + 1, stride=1, padding=0
        )

    def __call__(
        self, inputs: mx.array, input_lengths: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """Upsample inputs.

        Args:
            inputs: Input tensor (batch, time, channels) - MLX format.
            input_lengths: Input lengths (batch,).

        Returns:
            Upsampled tensor (batch, time * stride, out_channels) - MLX format.
            Updated lengths (batch,).
        """
        batch, time, channels = inputs.shape

        # Nearest neighbor interpolation on time axis
        outputs = mx.repeat(inputs, self.stride, axis=1)

        # Pad for convolution on time axis
        outputs = mx.pad(outputs, [(0, 0), (self.stride * 2, 0), (0, 0)])

        # Apply convolution
        outputs = self.conv(outputs)

        return outputs, input_lengths * self.stride


class PreLookaheadLayerMLX(nn.Module):
    """Pre-lookahead layer for MLX.

    Args:
        channels (int): Number of channels.
        pre_lookahead_len (int): Lookahead length.
    """

    def __init__(self, channels: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=pre_lookahead_len + 1,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=0,
        )

    def __call__(self, inputs: mx.array) -> mx.array:
        """Apply pre-lookahead.

        Args:
            inputs: Input tensor (batch, time, channels).

        Returns:
            Output tensor (batch, time, channels).
        """
        # MLX Conv1d expects (batch, time, channels) - no transpose needed
        outputs = inputs

        # Look ahead padding on time dimension (axis 1)
        outputs = mx.pad(outputs, [(0, 0), (0, self.pre_lookahead_len), (0, 0)])
        outputs = nn.leaky_relu(self.conv1(outputs))

        # Output conv - pad on time dimension
        outputs = mx.pad(outputs, [(0, 0), (2, 0), (0, 0)])
        outputs = self.conv2(outputs)

        # Residual connection
        outputs = outputs + inputs
        return outputs


class UpsampleConformerEncoderMLX(nn.Module):
    """Upsample Conformer Encoder for MLX.

    This encoder processes semantic tokens through Conformer layers,
    then upsamples and processes through additional layers.
    """

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 512,
        attention_heads: int = 8,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        input_layer: str = "linear",
        pos_enc_layer_type: str = "rel_pos_espnet",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        use_dynamic_left_chunk: bool = False,
        macaron_style: bool = False,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = False,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        key_bias: bool = True,
    ):
        """Construct an UpsampleConformerEncoderMLX object."""
        super().__init__()
        self._output_size = output_size
        self.normalize_before = normalize_before
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk

        # Create positional encoding
        if pos_enc_layer_type == "rel_pos_espnet":
            pos_enc = EspnetRelPositionalEncodingMLX(
                output_size, positional_dropout_rate
            )
        elif pos_enc_layer_type == "rel_pos":
            pos_enc = RelPositionalEncodingMLX(output_size, positional_dropout_rate)
        else:
            pos_enc = EspnetRelPositionalEncodingMLX(
                output_size, positional_dropout_rate
            )

        # Input embedding layer
        self.embed = LinearNoSubsamplingMLX(
            input_size, output_size, dropout_rate, pos_enc
        )

        # After norm
        self.after_norm = nn.LayerNorm(output_size, eps=1e-5)

        # Activation function
        if activation_type == "swish" or activation_type == "silu":
            activation = "swish"
        elif activation_type == "gelu":
            activation = "gelu"
        else:
            activation = "relu"

        # Build encoder layers with explicit attribute naming
        self.num_encoder_blocks = num_blocks
        for i in range(num_blocks):
            # Self-attention
            if selfattention_layer_type == "rel_selfattn":
                self_attn = RelPositionMultiHeadedAttentionMLX(
                    attention_heads, output_size, attention_dropout_rate, key_bias
                )
            else:
                self_attn = MultiHeadedAttentionMLX(
                    attention_heads, output_size, attention_dropout_rate, key_bias
                )

            # Feed-forward
            feed_forward = PositionwiseFeedForwardMLX(
                output_size, linear_units, dropout_rate, activation
            )

            # Macaron-style feed-forward
            feed_forward_macaron = None
            if macaron_style:
                feed_forward_macaron = PositionwiseFeedForwardMLX(
                    output_size, linear_units, dropout_rate, activation
                )

            # Convolution module
            conv_module = None
            if use_cnn_module:
                conv_module = ConvolutionModuleMLX(
                    output_size, cnn_module_kernel, activation, "layer_norm", causal
                )

            layer = ConformerEncoderLayerMLX(
                output_size,
                self_attn,
                feed_forward,
                feed_forward_macaron,
                conv_module,
                dropout_rate,
                normalize_before,
            )
            # Use explicit attribute naming for MLX parameter tracking
            setattr(self, f"encoders_{i}", layer)

        # Pre-lookahead layer
        self.pre_lookahead_layer = PreLookaheadLayerMLX(
            channels=512, pre_lookahead_len=3
        )

        # Upsampling layer
        self.up_layer = Upsample1DMLX(channels=512, out_channels=512, stride=2)

        # Create another positional encoding for upsampled sequence
        if pos_enc_layer_type == "rel_pos_espnet":
            up_pos_enc = EspnetRelPositionalEncodingMLX(
                output_size, positional_dropout_rate
            )
        else:
            up_pos_enc = RelPositionalEncodingMLX(output_size, positional_dropout_rate)

        self.up_embed = LinearNoSubsamplingMLX(
            input_size, output_size, dropout_rate, up_pos_enc
        )

        # Build upsampled encoder layers (4 layers as in original)
        self.num_up_encoder_blocks = 4
        for i in range(4):
            if selfattention_layer_type == "rel_selfattn":
                self_attn = RelPositionMultiHeadedAttentionMLX(
                    attention_heads, output_size, attention_dropout_rate, key_bias
                )
            else:
                self_attn = MultiHeadedAttentionMLX(
                    attention_heads, output_size, attention_dropout_rate, key_bias
                )

            feed_forward = PositionwiseFeedForwardMLX(
                output_size, linear_units, dropout_rate, activation
            )

            feed_forward_macaron = None
            if macaron_style:
                feed_forward_macaron = PositionwiseFeedForwardMLX(
                    output_size, linear_units, dropout_rate, activation
                )

            conv_module = None
            if use_cnn_module:
                conv_module = ConvolutionModuleMLX(
                    output_size, cnn_module_kernel, activation, "layer_norm", causal
                )

            layer = ConformerEncoderLayerMLX(
                output_size,
                self_attn,
                feed_forward,
                feed_forward_macaron,
                conv_module,
                dropout_rate,
                normalize_before,
            )
            # Use explicit attribute naming for MLX parameter tracking
            setattr(self, f"up_encoders_{i}", layer)

    def output_size(self) -> int:
        return self._output_size

    def __call__(
        self,
        xs: mx.array,
        xs_lens: mx.array,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[mx.array, mx.array]:
        """Encode input sequence.

        Args:
            xs: Input tensor (batch, time, dim).
            xs_lens: Input lengths (batch,).
            decoding_chunk_size: Chunk size for decoding.
            num_decoding_left_chunks: Number of left chunks.

        Returns:
            Encoded tensor (batch, time', dim).
            Output mask (batch, 1, time').
        """
        T = xs.shape[1]

        # Create mask
        masks = ~make_pad_mask(xs_lens, T)
        masks = mx.expand_dims(masks, axis=1)  # (B, 1, T)

        # Embed input
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks

        # Create chunk masks
        chunk_masks = add_optional_chunk_mask(
            xs,
            masks,
            self.use_dynamic_chunk,
            self.use_dynamic_left_chunk,
            decoding_chunk_size,
            self.static_chunk_size,
            num_decoding_left_chunks,
        )

        # Pre-lookahead layer
        xs = self.pre_lookahead_layer(xs)

        # Forward through encoder layers
        xs = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad)

        # Upsample - xs is already in MLX format (B, T, D)
        xs, xs_lens = self.up_layer(xs, xs_lens)

        T = xs.shape[1]
        masks = ~make_pad_mask(xs_lens, T)
        masks = mx.expand_dims(masks, axis=1)

        # Embed upsampled sequence
        xs, pos_emb, masks = self.up_embed(xs, masks)
        mask_pad = masks

        # Create chunk masks for upsampled sequence
        chunk_masks = add_optional_chunk_mask(
            xs,
            masks,
            self.use_dynamic_chunk,
            self.use_dynamic_left_chunk,
            decoding_chunk_size,
            self.static_chunk_size * self.up_layer.stride,
            num_decoding_left_chunks,
        )

        # Forward through upsampled encoder layers
        xs = self.forward_up_layers(xs, chunk_masks, pos_emb, mask_pad)

        # Final layer norm
        if self.normalize_before:
            xs = self.after_norm(xs)

        return xs, masks

    def forward_layers(
        self,
        xs: mx.array,
        chunk_masks: mx.array,
        pos_emb: mx.array,
        mask_pad: mx.array,
    ) -> mx.array:
        """Forward through encoder layers."""
        for i in range(self.num_encoder_blocks):
            layer = getattr(self, f"encoders_{i}")
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs

    def forward_up_layers(
        self,
        xs: mx.array,
        chunk_masks: mx.array,
        pos_emb: mx.array,
        mask_pad: mx.array,
    ) -> mx.array:
        """Forward through upsampled encoder layers."""
        for i in range(self.num_up_encoder_blocks):
            layer = getattr(self, f"up_encoders_{i}")
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs
