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
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
            "rms_norm_eps",
            "rope_theta",
            "attention_bias",
            "mlp_bias",
            "rope_scaling",
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
    """Rotary Position Embedding with Llama3-style scaling support."""

    def __init__(
        self,
        dims: int,
        max_position_embeddings: int = 131072,
        base: float = 500000.0,
        rope_scaling: Optional[Dict] = None,
    ):
        super().__init__()
        self.dims = dims
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.rope_scaling = rope_scaling

        # Compute base inverse frequencies
        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))

        # Apply Llama3-style scaling if configured
        if rope_scaling is not None and rope_scaling.get("rope_type") == "llama3":
            factor = rope_scaling.get("factor", 8.0)
            low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
            high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
            old_context_len = rope_scaling.get("original_max_position_embeddings", 8192)

            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor

            # Apply frequency-dependent scaling to get scaled inv_freq
            inv_freq_np = inv_freq.tolist()
            new_inv_freqs = []
            for freq in inv_freq_np:
                wavelen = 2 * 3.141592653589793 / freq
                if wavelen < high_freq_wavelen:
                    # High frequency: no scaling
                    new_inv_freqs.append(freq)
                elif wavelen > low_freq_wavelen:
                    # Low frequency: full scaling
                    new_inv_freqs.append(freq / factor)
                else:
                    # Mid frequency: linear interpolation
                    smooth = (old_context_len / wavelen - low_freq_factor) / (
                        high_freq_factor - low_freq_factor
                    )
                    new_inv_freqs.append((1 - smooth) * freq / factor + smooth * freq)

            # CRITICAL: mx.fast.rope 'freqs' param expects 1/inv_freq (regular frequencies),
            # not inv_freq (inverse frequencies). It internally computes inv_freq = 1/freqs.
            scaled_inv_freq = mx.array(new_inv_freqs, dtype=mx.float32)
            self._freqs = (
                1.0 / scaled_inv_freq
            )  # Convert inv_freq to freqs for mx.fast.rope
            self._use_custom_freqs = True
        else:
            self._freqs = None
            self._use_custom_freqs = False

    def __call__(
        self, q: mx.array, k: mx.array, offset: int = 0
    ) -> Tuple[mx.array, mx.array]:
        """
        Apply rotary position embeddings.

        Args:
            q: Query tensor of shape (B, n_heads, seq_len, head_dim)
            k: Key tensor of shape (B, n_kv_heads, seq_len, head_dim)
            offset: Position offset for cached sequences

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        if self._use_custom_freqs:
            # Use precomputed Llama3-scaled frequencies
            return (
                mx.fast.rope(
                    q,
                    self.dims,
                    traditional=False,
                    base=None,
                    scale=1.0,
                    offset=offset,
                    freqs=self._freqs,
                ),
                mx.fast.rope(
                    k,
                    self.dims,
                    traditional=False,
                    base=None,
                    scale=1.0,
                    offset=offset,
                    freqs=self._freqs,
                ),
            )
        else:
            # Standard RoPE
            return (
                mx.fast.rope(
                    q,
                    self.dims,
                    traditional=False,
                    base=self.base,
                    scale=1.0,
                    offset=offset,
                ),
                mx.fast.rope(
                    k,
                    self.dims,
                    traditional=False,
                    base=self.base,
                    scale=1.0,
                    offset=offset,
                ),
            )


class Attention(nn.Module):
    """Multi-head attention with optional GQA support."""

    def __init__(self, config: LlamaConfigMLX):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        # QKV projections
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        self.rope = RoPE(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        output_attentions: bool = False,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array], Optional[mx.array]]:
        """
        Forward pass for attention.

        Args:
            x: Input tensor of shape (B, L, D)
            mask: Optional attention mask
            cache: Optional (key, value) cache tuple
            output_attentions: Whether to return attention weights

        Returns:
            Tuple of (output, (new_k_cache, new_v_cache), attention_weights)
            attention_weights is None if output_attentions=False
        """
        B, L, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention: (B, L, n_heads, head_dim)
        q = mx.reshape(q, (B, L, self.num_heads, self.head_dim))
        k = mx.reshape(k, (B, L, self.num_kv_heads, self.head_dim))
        v = mx.reshape(v, (B, L, self.num_kv_heads, self.head_dim))

        # Transpose BEFORE RoPE: (B, n_heads, L, head_dim)
        # MLX fast.rope expects position on axis -2 (second-to-last)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        # Apply RoPE (now q/k are in correct shape for mx.fast.rope)
        offset = 0
        if cache is not None:
            k_cache, v_cache = cache
            offset = k_cache.shape[2]  # Position dim is now axis 2
        q, k = self.rope(q, k, offset=offset)

        # Handle KV cache (cache is already in transposed format)
        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate(
                [k_cache, k], axis=2
            )  # Concatenate on sequence dim (axis=2)
            v = mx.concatenate([v_cache, v], axis=2)

        # Grouped-query attention: repeat K, V if needed
        if self.num_heads != self.num_kv_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        # Compute attention
        attention_weights = None
        if output_attentions:
            # Manually compute attention for weight extraction
            # Q @ K^T: (B, n_heads, L_q, head_dim) @ (B, n_heads, head_dim, L_k) -> (B, n_heads, L_q, L_k)
            scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * self.scale

            # Apply causal mask
            L_q = q.shape[2]
            L_k = k.shape[2]
            # Create causal mask: upper triangular -inf
            causal_mask = mx.triu(mx.full((L_q, L_k), -1e9), k=L_k - L_q + 1)
            scores = scores + causal_mask[None, None, :, :]

            # Softmax
            attention_weights = mx.softmax(scores, axis=-1)  # (B, n_heads, L_q, L_k)

            # Apply attention to values
            output = mx.matmul(attention_weights, v)  # (B, n_heads, L_q, head_dim)
        else:
            # Use fast implementation (no attention weights)
            output = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask="causal"
            )

        # Reshape and project output
        output = mx.transpose(output, (0, 2, 1, 3))  # (B, L_q, n_heads, head_dim)
        output = mx.reshape(output, (B, L, -1))  # (B, L_q, n_heads * head_dim)
        output = self.o_proj(output)

        # Return output, updated cache, and optional attention weights
        # (k, v are in transposed format: B, n_heads, L, head_dim)
        return output, (k, v), attention_weights


class MLP(nn.Module):
    """Feed-forward network (SwiGLU)."""

    def __init__(self, config: LlamaConfigMLX):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )

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
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        output_attentions: bool = False,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array], Optional[mx.array]]:
        """
        Forward pass for transformer block.

        Args:
            x: Input tensor of shape (B, L, D)
            mask: Optional attention mask
            cache: Optional (key, value) cache
            output_attentions: Whether to return attention weights

        Returns:
            Tuple of (output, updated_cache, attention_weights)
        """
        # Self-attention with residual
        r, cache, attn_weights = self.self_attn(
            self.input_layernorm(x),
            mask=mask,
            cache=cache,
            output_attentions=output_attentions,
        )
        h = x + r

        # Feed-forward with residual
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r

        return out, cache, attn_weights


class LlamaModelMLX(nn.Module):
    """MLX implementation of Llama model."""

    def __init__(self, config: LlamaConfigMLX):
        super().__init__()
        self.config = config

        # Transformer layers
        self.layers = [
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ]
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
            output_attentions: Whether to return attention weights

        Returns:
            Dictionary with 'hidden_states', 'cache', and optionally 'attentions'
        """
        h = inputs_embeds

        # Initialize cache if needed
        if cache is None:
            cache = [None] * len(self.layers)

        new_cache = []
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None

        # Pass through transformer layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(h)

            h, layer_cache, attn_weights = layer(
                h, mask=mask, cache=cache[i], output_attentions=output_attentions
            )
            new_cache.append(layer_cache)

            if output_attentions:
                all_attentions.append(attn_weights)

        # Final layer norm
        h = self.norm(h)

        if output_hidden_states:
            all_hidden_states.append(h)

        result = {
            "hidden_states": all_hidden_states if output_hidden_states else [h],
            "cache": new_cache,
            "last_hidden_state": h,
        }

        if output_attentions:
            result["attentions"] = all_attentions

        return result
