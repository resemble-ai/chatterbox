# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
MLX implementation of Llama model for T3.
This is a simplified version focused on what T3 needs.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import mlx.core as mx
import mlx.nn as nn
import math


@dataclass
class LlamaConfigMLX:
    """MLX version of Llama configuration."""
    vocab_size: int = 8
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 64
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_scaling: Optional[Dict] = None

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary."""
        # Extract only the fields we need
        relevant_fields = {
            'vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
            'num_attention_heads', 'num_key_value_heads', 'head_dim',
            'max_position_embeddings', 'rms_norm_eps', 'rope_theta',
            'attention_bias', 'mlp_bias', 'rope_scaling'
        }
        filtered = {k: v for k, v in config_dict.items() if k in relevant_fields}
        return cls(**filtered)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class RoPE(nn.Module):
    """Rotary Position Embedding."""

    def __init__(self, dims: int, max_position_embeddings: int = 131072, base: float = 500000.0,
                 rope_scaling: Optional[Dict] = None):
        super().__init__()
        self.dims = dims
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Handle rope scaling (Llama 3 style)
        if rope_scaling is not None and rope_scaling.get('rope_type') == 'llama3':
            self.factor = rope_scaling.get('factor', 8.0)
            self.low_freq_factor = rope_scaling.get('low_freq_factor', 1.0)
            self.high_freq_factor = rope_scaling.get('high_freq_factor', 4.0)
            self.original_max_position = rope_scaling.get('original_max_position_embeddings', 8192)
        else:
            self.factor = 1.0
            self.low_freq_factor = 1.0
            self.high_freq_factor = 1.0
            self.original_max_position = max_position_embeddings

    def __call__(self, q: mx.array, k: mx.array, offset: int = 0) -> Tuple[mx.array, mx.array]:
        """
        Apply rotary position embeddings.

        Args:
            q: Query tensor of shape (..., seq_len, n_heads, head_dim)
            k: Key tensor of shape (..., seq_len, n_heads, head_dim)
            offset: Position offset for cached sequences

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        # Use MLX's RoPE implementation with required parameters
        # traditional=False for Llama-style RoPE
        return mx.fast.rope(q, self.dims, traditional=False, base=self.base, scale=1.0, offset=offset), \
               mx.fast.rope(k, self.dims, traditional=False, base=self.base, scale=1.0, offset=offset)


class Attention(nn.Module):
    """Multi-head attention with optional GQA support."""

    def __init__(self, config: LlamaConfigMLX):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5

        # QKV projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rope = RoPE(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        """
        Forward pass for attention.

        Args:
            x: Input tensor of shape (B, L, D)
            mask: Optional attention mask
            cache: Optional (key, value) cache tuple

        Returns:
            Tuple of (output, (new_k_cache, new_v_cache))
        """
        B, L, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = mx.reshape(q, (B, L, self.num_heads, self.head_dim))
        k = mx.reshape(k, (B, L, self.num_kv_heads, self.head_dim))
        v = mx.reshape(v, (B, L, self.num_kv_heads, self.head_dim))

        # Apply RoPE
        offset = 0
        if cache is not None:
            k_cache, v_cache = cache
            offset = k_cache.shape[1] if k_cache.ndim == 4 else k_cache.shape[2]
        q, k = self.rope(q, k, offset=offset)

        # Transpose for attention BEFORE caching: (B, n_heads, L, head_dim)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        # Handle KV cache (cache is already in transposed format)
        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=2)  # Concatenate on sequence dim (axis=2)
            v = mx.concatenate([v_cache, v], axis=2)

        # Grouped-query attention: repeat K, V if needed
        if self.num_heads != self.num_kv_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        # Compute attention scores
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) * self.scale

        if mask is not None:
            scores = scores + mask

        # Compute attention weights and output
        attn_weights = mx.softmax(scores, axis=-1)
        output = attn_weights @ v  # (B, n_heads, L_q, head_dim)

        # Reshape and project output
        output = mx.transpose(output, (0, 2, 1, 3))  # (B, L_q, n_heads, head_dim)
        output = mx.reshape(output, (B, L, -1))  # (B, L_q, n_heads * head_dim)
        output = self.o_proj(output)

        # Return output and updated cache (k, v are in transposed format: B, n_heads, L, head_dim)
        return output, (k, v)


class MLP(nn.Module):
    """Feed-forward network (SwiGLU)."""

    def __init__(self, config: LlamaConfigMLX):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def __call__(self, x: mx.array) -> mx.array:
        # SwiGLU activation
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Single Llama transformer block."""

    def __init__(self, config: LlamaConfigMLX):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        """
        Forward pass for transformer block.

        Args:
            x: Input tensor of shape (B, L, D)
            mask: Optional attention mask
            cache: Optional (key, value) cache

        Returns:
            Tuple of (output, updated_cache)
        """
        # Self-attention with residual
        r, cache = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        h = x + r

        # Feed-forward with residual
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r

        return out, cache


class LlamaModelMLX(nn.Module):
    """MLX implementation of Llama model."""

    def __init__(self, config: LlamaConfigMLX):
        super().__init__()
        self.config = config

        # Transformer layers
        self.layers = [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs_embeds: mx.array,
        cache: Optional[list] = None,
        mask: Optional[mx.array] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Dict:
        """
        Forward pass for Llama model.

        Args:
            inputs_embeds: Input embeddings of shape (B, L, D)
            cache: Optional list of (key, value) caches for each layer
            mask: Optional attention mask
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights (not implemented)

        Returns:
            Dictionary with 'hidden_states' and optionally 'cache'
        """
        h = inputs_embeds

        # Initialize cache if needed
        if cache is None:
            cache = [None] * len(self.layers)

        new_cache = []
        all_hidden_states = [] if output_hidden_states else None

        # Pass through transformer layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(h)

            h, layer_cache = layer(h, mask=mask, cache=cache[i])
            new_cache.append(layer_cache)

        # Final layer norm
        h = self.norm(h)

        if output_hidden_states:
            all_hidden_states.append(h)

        return {
            'hidden_states': all_hidden_states if output_hidden_states else [h],
            'cache': new_cache,
            'last_hidden_state': h,
        }
