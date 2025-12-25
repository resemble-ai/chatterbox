from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .convolution import ConvolutionModule
from .encoder_layer import ConformerEncoderLayer
from .positionwise_feed_forward import PositionwiseFeedForward
from ..utils.class_utils import (
    COSYVOICE_EMB_CLASSES,
    COSYVOICE_SUBSAMPLE_CLASSES,
    COSYVOICE_ATTENTION_CLASSES,
    COSYVOICE_ACTIVATION_CLASSES,
)
from ..utils.mask import make_pad_mask
from ..utils.mask import add_optional_chunk_mask


class Upsample1D(nn.Module):

    def __init__(self, channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv = nn.Conv1d(self.channels, self.out_channels, stride * 2 + 1, stride=1, padding=0)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor):
        outputs = F.interpolate(inputs, scale_factor=float(self.stride), mode="nearest")
        outputs = F.pad(outputs, (self.stride * 2, 0), value=0.0)
        outputs = self.conv(outputs)
        return outputs, input_lengths * self.stride


class PreLookaheadLayer(nn.Module):
    def __init__(self, channels: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=pre_lookahead_len + 1,
            stride=1, padding=0,
        )
        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=3, stride=1, padding=0,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs.transpose(1, 2).contiguous()
        outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode='constant', value=0.0)
        outputs = F.leaky_relu(self.conv1(outputs))
        outputs = F.pad(outputs, (2, 0), mode='constant', value=0.0)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()
        outputs = outputs + inputs
        return outputs


class UpsampleConformerEncoder(torch.nn.Module):

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
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = False,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = False,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        key_bias: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self._output_size = output_size

        self.global_cmvn = global_cmvn
        self.embed = COSYVOICE_SUBSAMPLE_CLASSES[input_layer](
            input_size,
            output_size,
            dropout_rate,
            COSYVOICE_EMB_CLASSES[pos_enc_layer_type](output_size,
                                                      positional_dropout_rate),
        )

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.gradient_checkpointing = gradient_checkpointing
        activation = COSYVOICE_ACTIVATION_CLASSES[activation_type]()
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            key_bias,
        )
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        convolution_layer_args = (output_size, cnn_module_kernel, activation,
                                  cnn_module_norm, causal)
        self.pre_lookahead_layer = PreLookaheadLayer(channels=512, pre_lookahead_len=3)
        self.encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                COSYVOICE_ATTENTION_CLASSES[selfattention_layer_type](
                    *encoder_selfattn_layer_args),
                PositionwiseFeedForward(*positionwise_layer_args),
                PositionwiseFeedForward(
                    *positionwise_layer_args) if macaron_style else None,
                ConvolutionModule(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
            ) for _ in range(num_blocks)
        ])
        self.up_layer = Upsample1D(channels=512, out_channels=512, stride=2)
        self.up_embed = COSYVOICE_SUBSAMPLE_CLASSES[input_layer](
            input_size,
            output_size,
            dropout_rate,
            COSYVOICE_EMB_CLASSES[pos_enc_layer_type](output_size,
                                                      positional_dropout_rate),
        )
        self.up_encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                COSYVOICE_ATTENTION_CLASSES[selfattention_layer_type](
                    *encoder_selfattn_layer_args),
                PositionwiseFeedForward(*positionwise_layer_args),
                PositionwiseFeedForward(
                    *positionwise_layer_args) if macaron_style else None,
                ConvolutionModule(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
            ) for _ in range(4)
        ])

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks
        chunk_masks = add_optional_chunk_mask(xs, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size,
                                              num_decoding_left_chunks)
        xs = self.pre_lookahead_layer(xs)
        xs = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad)

        xs = xs.transpose(1, 2).contiguous()
        xs, xs_lens = self.up_layer(xs, xs_lens)
        xs = xs.transpose(1, 2).contiguous()
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)
        xs, pos_emb, masks = self.up_embed(xs, masks)
        mask_pad = masks
        chunk_masks = add_optional_chunk_mask(xs, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size * self.up_layer.stride,
                                              num_decoding_left_chunks)
        xs = self.forward_up_layers(xs, chunk_masks, pos_emb, mask_pad)

        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks

    def forward_layers(self, xs: torch.Tensor, chunk_masks: torch.Tensor,
                       pos_emb: torch.Tensor,
                       mask_pad: torch.Tensor) -> torch.Tensor:
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs

    def forward_up_layers(self, xs: torch.Tensor, chunk_masks: torch.Tensor,
                          pos_emb: torch.Tensor,
                          mask_pad: torch.Tensor) -> torch.Tensor:
        for layer in self.up_encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs