# Copyright (c) 2025 MichaelYangAI
# MIT License

import math
import mlx.core as mx
import mlx.nn as nn


class AttentionQKVMLX(nn.Module):
    """
    MLX implementation of multi-head attention with separate Q, K, V projections.
    """

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        dropout_rate: float = 0.1,
        scale: float = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = scale if scale is not None else head_dim**-0.5
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

    def __call__(
        self, q: mx.array, k: mx.array, v: mx.array, mask: mx.array = None
    ) -> mx.array:
        """
        Forward pass for attention.

        Args:
            q, k, v: Query, key, value tensors of shape (B, L, D)
            mask: Optional attention mask

        Returns:
            Attention output of shape (B, L, D)
        """
        q, k, v = [self.split_heads(tensor) for tensor in [q, k, v]]
        out = self.scaled_dot_product_attention(q, k, v, mask=mask)
        return self.combine_heads(out)

    def scaled_dot_product_attention(
        self, q: mx.array, k: mx.array, v: mx.array, mask: mx.array = None
    ) -> mx.array:
        """
        Compute scaled dot-product attention.

        Args:
            q: Query tensor of shape (B, H, L, D)
            k: Key tensor of shape (B, H, S, D)
            v: Value tensor of shape (B, H, S, D)
            mask: Optional mask

        Returns:
            Attention output of shape (B, H, L, D)
        """
        # Compute attention scores: (B, H, L, S)
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) * self.scale

        if mask is not None:
            scores = mx.where(mask == 0, -1e9, scores)

        # Compute attention weights
        attn = mx.softmax(scores, axis=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        return attn @ v  # (B, H, L, D)

    def split_heads(self, x: mx.array) -> mx.array:
        """
        Split tensor into attention heads.

        Args:
            x: Input tensor of shape (B, L, n_heads * head_dim)

        Returns:
            Reshaped tensor of shape (B, n_heads, L, head_dim)
        """
        bs, length, _ = x.shape
        x = mx.reshape(x, (bs, length, self.n_heads, self.head_dim))
        return mx.transpose(x, (0, 2, 1, 3))  # (B, H, L, D)

    def combine_heads(self, x: mx.array) -> mx.array:
        """
        Combine attention heads.

        Args:
            x: Input tensor of shape (B, H, L, D)

        Returns:
            Combined tensor of shape (B, L, H*D)
        """
        bs, _, length, _ = x.shape
        x = mx.transpose(x, (0, 2, 1, 3))  # (B, L, H, D)
        return mx.reshape(x, (bs, length, -1))


class AttentionBlock2MLX(nn.Module):
    """
    MLX implementation of cross-attention block with separate Q, K, V projections.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
        dropout_rate: float = 0.2,
        scale: float = None,
    ):
        super().__init__()
        self.channels = channels

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.norm = nn.LayerNorm(channels)

        # Separate linear layers for Q, K, V
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)

        self.attention = AttentionQKVMLX(
            self.num_heads,
            channels // self.num_heads,
            dropout_rate=dropout_rate,
            scale=scale,
        )

        self.proj_out = nn.Linear(channels, channels)

    def __call__(self, x1: mx.array, x2: mx.array, mask: mx.array = None) -> mx.array:
        """
        Forward pass for cross-attention.

        Args:
            x1: Query input of shape (B, L1, C)
            x2: Key/Value input of shape (B, L2, C)
            mask: Optional attention mask

        Returns:
            Output of shape (B, L1, C)
        """
        b1, l1, c1 = x1.shape

        # Normalize inputs
        x1_norm = self.norm(x1)
        x2_norm = self.norm(x2)

        # Compute Q, K, V
        q = self.to_q(x1_norm)
        k = self.to_k(x2_norm)
        v = self.to_v(x2_norm)

        # Apply attention
        h = self.attention(q, k, v, mask=mask)
        h = self.proj_out(h)

        # Residual connection
        return x1 + h


class PerceiverMLX(nn.Module):
    """
    MLX implementation of Perceiver resampler.
    Inspired by https://arxiv.org/abs/2103.03206
    Converted from PyTorch version in perceiver.py
    """

    def __init__(
        self,
        pre_attention_query_token: int = 32,
        pre_attention_query_size: int = 1024,
        embedding_dim: int = 1024,
        num_attn_heads: int = 4,
    ):
        """
        Initialize the Perceiver module.

        Args:
            pre_attention_query_token: Number of query tokens for pre-attention
            pre_attention_query_size: Size of each query token
            embedding_dim: Dimension of the embedding space
            num_attn_heads: Number of attention heads
        """
        super().__init__()

        # Calculate variance for uniform initialization
        query_variance = math.sqrt(3.0) * math.sqrt(
            2.0 / (pre_attention_query_token + pre_attention_query_token)
        )

        # Initialize learnable query tokens as a proper parameter
        # Store directly on self so MLX load_weights can find it
        self.pre_attention_query = mx.random.uniform(
            low=-query_variance,
            high=query_variance,
            shape=(1, pre_attention_query_token, pre_attention_query_size),
        )

        # Attention block
        self.attn = AttentionBlock2MLX(embedding_dim, num_attn_heads)

    def __call__(self, h: mx.array) -> mx.array:
        """
        Forward pass of the Perceiver module.

        Args:
            h: Input tensor of shape (B, L, D)

        Returns:
            Resampled output of shape (B, pre_attention_query_token, D)
        """
        batch_size = h.shape[0]

        # Expand query to match batch size
        query_ = mx.broadcast_to(
            self.pre_attention_query,
            (
                batch_size,
                self.pre_attention_query.shape[1],
                self.pre_attention_query.shape[2],
            ),
        )

        # Cross-attention: query attends to input
        pre_att = self.attn(query_, h)

        # Self-attention: refine the queries
        attn = self.attn(pre_att, pre_att)

        return attn
