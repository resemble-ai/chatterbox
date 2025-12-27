# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
# MLX port for Apple Silicon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Flow Module MLX Implementation

Masked Diffusion with XVector conditioning for speech synthesis.
Combines encoder, flow matching decoder, and speaker conditioning.
"""

import logging
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .utils.mask_mlx import make_pad_mask

logger = logging.getLogger(__name__)


class MaskedDiffWithXvecMLX(nn.Module):
    """
    Masked Diffusion with XVector conditioning (non-causal version).

    This module combines:
    - Input embedding for speech tokens
    - Conformer encoder for feature extraction
    - Flow matching decoder for mel generation
    - Speaker embedding conditioning via xvector
    - Optional length regulator for duration control
    """

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        vocab_size: int = 4096,
        input_frame_rate: int = 50,
        encoder: nn.Module = None,
        length_regulator: nn.Module = None,
        decoder: nn.Module = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.input_frame_rate = input_frame_rate

        # Token embedding
        self.input_embedding = nn.Embedding(vocab_size, input_size)

        # Speaker embedding projection
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size)

        # Encoder
        self.encoder = encoder
        if encoder is not None:
            self.encoder_proj = nn.Linear(encoder.output_size(), output_size)
        else:
            self.encoder_proj = None

        # Decoder (CFM)
        self.decoder = decoder

        # Length regulator (optional)
        self.length_regulator = length_regulator

    def __call__(
        self,
        token: mx.array,
        token_len: mx.array,
        prompt_token: mx.array,
        prompt_token_len: int,
        prompt_feat: mx.array,
        prompt_feat_len: int,
        embedding: mx.array,
        flow_cache: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Inference forward pass.

        Args:
            token: Speech tokens [B, T_tok]
            token_len: Token lengths [B]
            prompt_token: Prompt tokens [B, T_prompt]
            prompt_token_len: Prompt token length
            prompt_feat: Prompt mel features [B, T_mel, C]
            prompt_feat_len: Prompt mel length
            embedding: Speaker embedding [B, spk_dim]
            flow_cache: Optional cache for streaming

        Returns:
            Generated mel spectrogram [B, C, T]
            Updated flow cache
        """
        assert token.shape[0] == 1, "Only batch size 1 supported"

        # Normalize and project speaker embedding
        embedding = embedding / (
            mx.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8
        )
        embedding = self.spk_embed_affine_layer(embedding)

        # Concatenate prompt and target tokens
        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
        token = mx.concatenate([prompt_token, token], axis=1)
        token_len = prompt_token_len + token_len

        # Create mask and embed tokens
        mask = ~make_pad_mask(token_len, max_len=token.shape[1])
        mask = mx.expand_dims(mask, axis=-1).astype(embedding.dtype)

        # Clamp token IDs to valid range
        token = mx.clip(token, 0, self.vocab_size - 1)
        token_embedded = self.input_embedding(token) * mask

        # Encode tokens
        h, h_lengths = self.encoder(token_embedded, token_len)
        h = self.encoder_proj(h)

        # Length regulation
        mel_len1, mel_len2 = prompt_feat.shape[1], int(
            token_len2 / self.input_frame_rate * 22050 / 256
        )
        h, h_lengths = self.length_regulator.inference(
            h[:, :token_len1],
            h[:, token_len1:],
            mel_len1,
            mel_len2,
            self.input_frame_rate,
        )

        # Prepare conditions - concatenate prompt_feat with zeros
        zeros_part = mx.zeros([1, mel_len2, self.output_size], dtype=h.dtype)
        conds = mx.concatenate([prompt_feat, zeros_part], axis=1)  # [B, T, C]
        conds = mx.transpose(conds, axes=(0, 2, 1))  # [B, C, T]

        # Create mel mask
        mel_mask = ~make_pad_mask(
            mx.array([mel_len1 + mel_len2]), max_len=mel_len1 + mel_len2
        )
        mel_mask = mel_mask.astype(h.dtype)

        # Run decoder
        feat, flow_cache = self.decoder(
            mu=mx.transpose(h, axes=(0, 2, 1)),  # [B, C, T]
            mask=mx.expand_dims(mel_mask, axis=1),
            spks=embedding,
            cond=conds,
            n_timesteps=5,
            prompt_len=mel_len1,
            flow_cache=flow_cache,
        )

        # Extract generated portion (after prompt)
        feat = feat[:, :, mel_len1:]

        return feat.astype(mx.float32), flow_cache


class CausalMaskedDiffWithXvecMLX(nn.Module):
    """
    Causal Masked Diffusion with XVector conditioning.

    Similar to MaskedDiffWithXvecMLX but uses causal attention
    for streaming inference.
    """

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        vocab_size: int = 6561,
        input_frame_rate: int = 25,
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.input_frame_rate = input_frame_rate
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len

        # Token embedding
        self.input_embedding = nn.Embedding(vocab_size, input_size)

        # Speaker embedding projection
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size)

        # Encoder
        self.encoder = encoder
        if encoder is not None:
            self.encoder_proj = nn.Linear(encoder.output_size(), output_size)
        else:
            self.encoder_proj = None

        # Decoder (CFM)
        self.decoder = decoder

    def __call__(
        self,
        token: mx.array,
        token_len: mx.array,
        prompt_token: mx.array,
        prompt_token_len: int,
        prompt_feat: mx.array,
        prompt_feat_len: Optional[int],
        embedding: mx.array,
        finalize: bool = False,
    ) -> Tuple[mx.array, None]:
        """
        Inference forward pass for causal model.

        Args:
            token: Speech tokens [B, T_tok]
            token_len: Token lengths [B]
            prompt_token: Prompt tokens [B, T_prompt]
            prompt_token_len: Prompt token length
            prompt_feat: Prompt mel features [B, T_mel, C]
            prompt_feat_len: Prompt mel length (unused)
            embedding: Speaker embedding [B, spk_dim]
            finalize: Whether this is the final chunk

        Returns:
            Generated mel spectrogram [B, C, T]
            None (no cache in causal mode)
        """
        assert token.shape[0] == 1, "Only batch size 1 supported"

        # Normalize and project speaker embedding
        embedding = embedding / (
            mx.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8
        )
        embedding = self.spk_embed_affine_layer(embedding)

        # Concatenate prompt and target tokens
        token = mx.concatenate([prompt_token, token], axis=1)
        token_len = prompt_token_len + token_len

        # Create mask and embed tokens
        mask = ~make_pad_mask(token_len, max_len=token.shape[1])
        mask = mx.expand_dims(mask, axis=-1).astype(embedding.dtype)

        # Clamp token IDs to valid range
        token = mx.clip(token, 0, self.vocab_size - 1)
        token_embedded = self.input_embedding(token) * mask

        # Encode tokens
        h, h_lengths = self.encoder(token_embedded, token_len)

        # Handle lookahead for non-final chunks
        if not finalize:
            h = h[:, : -self.pre_lookahead_len * self.token_mel_ratio]

        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
        h = self.encoder_proj(h)

        # Prepare conditions - concatenate prompt_feat with zeros
        zeros_part = mx.zeros([1, mel_len2, self.output_size], dtype=h.dtype)
        conds = mx.concatenate([prompt_feat, zeros_part], axis=1)  # [B, T, C]
        conds = mx.transpose(conds, axes=(0, 2, 1))  # [B, C, T]

        # Create mel mask
        mel_mask = ~make_pad_mask(
            mx.array([mel_len1 + mel_len2]), max_len=mel_len1 + mel_len2
        )
        mel_mask = mel_mask.astype(h.dtype)

        # Run decoder (5 midpoint steps = 10 function evaluations)
        feat, _ = self.decoder(
            mu=mx.transpose(h, axes=(0, 2, 1)),  # [B, C, T]
            mask=mx.expand_dims(mel_mask, axis=1),
            spks=embedding,
            cond=conds,
            n_timesteps=5,
        )

        # Extract generated portion (after prompt)
        feat = feat[:, :, mel_len1:]

        return feat.astype(mx.float32), None


def convert_flow_weights(pytorch_state_dict: dict) -> dict:
    """
    Convert PyTorch flow module weights to MLX format.

    Args:
        pytorch_state_dict: PyTorch state dict

    Returns:
        Dictionary with MLX-compatible weights
    """
    mlx_weights = {}

    for key, value in pytorch_state_dict.items():
        numpy_val = value.numpy() if hasattr(value, "numpy") else value

        # Handle embedding
        if "input_embedding" in key:
            mlx_weights[key] = mx.array(numpy_val)

        # Handle linear layers
        elif "spk_embed_affine_layer" in key or "encoder_proj" in key:
            if "weight" in key:
                # Linear weight stays [out, in]
                mlx_weights[key] = mx.array(numpy_val)
            elif "bias" in key:
                mlx_weights[key] = mx.array(numpy_val)

        # Encoder and decoder weights are handled separately
        elif "encoder." in key or "decoder." in key:
            # These will be converted by their respective converters
            mlx_weights[key] = mx.array(numpy_val)

        else:
            mlx_weights[key] = mx.array(numpy_val)

    return mlx_weights
