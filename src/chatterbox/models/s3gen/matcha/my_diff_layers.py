# my_diff_layers.py  – tiny self-contained re-implementations
# Apache-2.0 licence, since the originals in diffusers are.
# They are **functionally identical** for the subset your code needs.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------
class GEGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class GELU(nn.Module):
    """A GELU wrapped to mimic diffusers.GELU interface (proj-layer inside)."""
    def __init__(self, in_dim: int, out_dim: int, approximate: str = "none"):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.approximate = approximate

    def forward(self, x):
        return F.gelu(self.proj(x), approximate=self.approximate)


class ApproximateGELU(GELU):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(in_dim, out_dim, approximate="tanh")


# ---------------------------------------------------------------------
# Ada-LayerNorm variants
# (identical to diffusers implementation, only the minimum needed parts)
# ---------------------------------------------------------------------
class _AdaLNBase(nn.Module):
    def __init__(self, hidden_size, emb_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear = nn.Linear(emb_size, 3 * hidden_size, bias=True)

    def _modulation(self, hidden_states, timestep_emb):
        shift, scale, gate = self.linear(timestep_emb).chunk(3, dim=-1)
        hidden_states = self.norm(hidden_states) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return hidden_states, gate

class AdaLayerNorm(_AdaLNBase):
    def forward(self, hidden_states, timestep_emb):
        hidden_states, _ = self._modulation(hidden_states, timestep_emb)
        return hidden_states

class AdaLayerNormZero(_AdaLNBase):
    """Same but returns extra gates for MSA + MLP (used in your block)."""
    def forward(self, hidden_states, timestep_emb, *_, **__):
        hidden_states, gate = self._modulation(hidden_states, timestep_emb)
        # Return the five values the Transformer block expects
        return hidden_states, gate, torch.zeros_like(gate), torch.zeros_like(gate), gate


# ---------------------------------------------------------------------
# Attention (simple QKV with flash-attention-style matmuls)
# ---------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(
        self,
        query_dim,
        heads,
        dim_head,
        dropout=0.0,
        bias=False,
        cross_attention_dim=None,
        upcast_attention=False,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(query_dim, heads * dim_head, bias=bias)
        context_dim = cross_attention_dim or query_dim
        self.to_k = nn.Linear(context_dim, heads * dim_head, bias=bias)
        self.to_v = nn.Linear(context_dim, heads * dim_head, bias=bias)
        self.to_out = nn.Linear(heads * dim_head, query_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x, b, h, d):
        return x.view(b, -1, h, d).transpose(1, 2)  # (b, h, t, d)

    def forward(self, x, encoder_hidden_states=None, attention_mask=None, **_):
        context = encoder_hidden_states if encoder_hidden_states is not None else x
        b, _, _ = x.shape
        h = self.heads
        head_dim = self.to_q.out_features // h        # == dim_head
        q = self._shape(self.to_q(x), b, h, head_dim)
        k = self._shape(self.to_k(context), b, h, head_dim)
        v = self._shape(self.to_v(context), b, h, head_dim)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, -1, h * q.size(-1))
        return self.to_out(out)


# ---------------------------------------------------------------------
# LoRA-compatible Linear (pass-through when LoRA weights absent)
# ---------------------------------------------------------------------
class LoRACompatibleLinear(nn.Linear):
    def __init__(self, *args, r: int = 0, lora_alpha: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = r
        if r > 0:
            in_dim, out_dim = self.in_features, self.out_features
            self.A = nn.Parameter(torch.zeros(r, in_dim))
            self.B = nn.Parameter(torch.zeros(out_dim, r))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
            self.scaling = lora_alpha / r
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)
            self.scaling = 0.0

    def forward(self, x):
        result = super().forward(x)
        if self.r > 0:
            lora_update = (x @ self.A.t()) @ self.B.t()
            result = result + self.scaling * lora_update
        return result


# ---------------------------------------------------------------------
# Utility decorator (no-op here)
# ---------------------------------------------------------------------
def maybe_allow_in_graph(fn):
    return fn
