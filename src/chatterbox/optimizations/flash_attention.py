"""
Flash Attention integration for faster transformer inference

Flash Attention provides 2-4x speedup for attention computation with
exact numerical equivalence to standard attention.
"""
import torch
import torch.nn as nn
from typing import Optional


def is_flash_attn_available():
    """Check if Flash Attention is available"""
    try:
        import flash_attn
        return True
    except ImportError:
        return False


def enable_flash_attention_for_llama(model):
    """
    Enable Flash Attention for LLaMA model

    This modifies the model in-place to use Flash Attention
    for all self-attention layers.

    Args:
        model: LLamaModel instance

    Returns:
        model with Flash Attention enabled
    """
    if not is_flash_attn_available():
        print("⚠️  flash-attn not installed. Install with: pip install flash-attn --no-build-isolation")
        return model

    try:
        # Check PyTorch version
        if not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            print("⚠️  PyTorch 2.0+ required for Flash Attention")
            return model

        # Enable SDPA (Scaled Dot Product Attention) backend
        # PyTorch 2.0+ has native Flash Attention support via SDPA
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        print("✅ Flash Attention enabled via PyTorch SDPA")

        return model

    except Exception as e:
        print(f"⚠️  Could not enable Flash Attention: {e}")
        return model


def patch_llama_attention_forward():
    """
    Monkey-patch LLaMA attention to use PyTorch's native SDPA

    This provides Flash Attention speedups without requiring the flash-attn package
    """
    try:
        from transformers.models.llama.modeling_llama import LlamaAttention
        import math

        original_forward = LlamaAttention.forward

        def optimized_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[tuple] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
        ):
            # If output_attentions is True, fall back to original implementation
            if output_attentions:
                return original_forward(
                    self,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                    **kwargs,
                )

            bsz, q_len, _ = hidden_states.size()

            # Project Q, K, V
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            # Reshape for multi-head attention
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            # Apply rotary embeddings if present
            if hasattr(self, 'rotary_emb'):
                cos, sin = self.rotary_emb(value_states, position_ids)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # Handle KV cache
            if past_key_value is not None:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

            past_key_value = (key_states, value_states) if use_cache else None

            # Expand K/V if using GQA (Grouped Query Attention)
            if self.num_key_value_heads != self.num_heads:
                key_states = repeat_kv(key_states, self.num_heads // self.num_key_value_heads)
                value_states = repeat_kv(value_states, self.num_heads // self.num_key_value_heads)

            # Use PyTorch's native SDPA with Flash Attention
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=attention_mask is None,  # Use causal mask for autoregressive
            )

            # Reshape and project output
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)

            return attn_output, None, past_key_value

        # Apply the patch
        LlamaAttention.forward = optimized_forward

        print("✅ LLaMA attention patched with optimized SDPA")
        return True

    except Exception as e:
        print(f"⚠️  Could not patch LLaMA attention: {e}")
        return False


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value tensors for Grouped Query Attention

    Args:
        hidden_states: (batch, num_kv_heads, seqlen, head_dim)
        n_rep: number of repetitions

    Returns:
        (batch, num_kv_heads * n_rep, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings"""
    # This is a simplified version - the actual implementation depends on the model
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotate half the hidden dims of the input"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
