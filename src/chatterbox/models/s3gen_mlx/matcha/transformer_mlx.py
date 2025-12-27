# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
MLX implementation of BasicTransformerBlock for Matcha decoder.
Port of PyTorch implementation from s3gen/matcha/transformer.py
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class FeedForwardMLX(nn.Module):
    """Feed-forward layer with activation.

    Args:
        dim: Input/output dimension.
        dim_out: Output dimension (defaults to dim).
        mult: Multiplier for hidden dimension.
        dropout: Dropout rate (not used in inference).
        activation_fn: Activation function name.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        # GEGLU splits the projection in half for gating
        if activation_fn in ["geglu", "geglu-approximate"]:
            # Project to 2x inner_dim for GEGLU
            self.proj_in = nn.Linear(dim, inner_dim * 2)
            self.use_geglu = True
        else:
            self.proj_in = nn.Linear(dim, inner_dim)
            self.use_geglu = False
            if activation_fn == "gelu" or activation_fn == "gelu-approximate":
                self.act = nn.GELU()
            else:
                self.act = nn.SiLU()

        self.proj_out = nn.Linear(inner_dim, dim_out)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        if self.use_geglu:
            hidden_states = self.proj_in(hidden_states)
            # GEGLU: split and gate
            hidden_states, gate = mx.split(hidden_states, 2, axis=-1)
            hidden_states = hidden_states * nn.gelu(gate)
        else:
            hidden_states = self.proj_in(hidden_states)
            hidden_states = self.act(hidden_states)

        return self.proj_out(hidden_states)


class AttentionMLX(nn.Module):
    """Self-attention layer for MLX.

    Args:
        query_dim: Query dimension.
        heads: Number of attention heads.
        dim_head: Dimension per head.
        dropout: Dropout rate.
        bias: Whether to use bias.
    """

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = True,  # Match PyTorch default
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, query_dim, bias=bias)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            hidden_states: Input (batch, seq, dim).
            attention_mask: Attention mask.

        Returns:
            Output (batch, seq, dim).
        """
        batch, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)

        # Reshape for multi-head attention
        q = mx.reshape(q, (batch, seq_len, self.heads, self.dim_head))
        k = mx.reshape(k, (batch, seq_len, self.heads, self.dim_head))
        v = mx.reshape(v, (batch, seq_len, self.heads, self.dim_head))

        # Transpose to (batch, heads, seq, dim_head)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        # Compute attention scores
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) * self.scale

        # Apply attention mask (already in bias form)
        if attention_mask is not None:
            # Ensure mask broadcasts correctly to [B, heads, T, T]
            # If mask is [B, 1, T], expand to [B, 1, 1, T]
            if attention_mask.ndim == 3:
                attention_mask = mx.expand_dims(attention_mask, axis=2)
            scores = scores + attention_mask

        # Softmax in FP32 for stability
        attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)

        # Apply attention to values
        out = attn @ v

        # Reshape back
        out = mx.transpose(out, (0, 2, 1, 3))
        out = mx.reshape(out, (batch, seq_len, -1))

        return self.to_out(out)


class BasicTransformerBlockMLX(nn.Module):
    """Basic transformer block for MLX.

    Args:
        dim: Input dimension.
        num_attention_heads: Number of attention heads.
        attention_head_dim: Dimension per head.
        dropout: Dropout rate.
        activation_fn: Activation function for feed-forward.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
    ):
        super().__init__()

        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = AttentionMLX(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
        )

        # Feed-forward
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForwardMLX(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        timestep: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            hidden_states: Input (batch, seq, dim).
            attention_mask: Attention mask (already in bias form).
            timestep: Timestep embedding (not used in basic block).

        Returns:
            Output (batch, seq, dim).
        """
        # Self-attention with residual
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, attention_mask)
        hidden_states = attn_output + hidden_states

        # Feed-forward with residual
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states

        return hidden_states
