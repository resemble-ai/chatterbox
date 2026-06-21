# ADR-0002: Backbone Construction Strategy

**Status:** Accepted  
**Date:** 2026-06-21

## Context

The T3 backbone is a LLaMA-3-style decoder-only transformer: RMSNorm → RoPE-attention → RMSNorm → SwiGLU-MLP × 30 layers. MLX provides `nn.TransformerDecoderLayer` but it uses LayerNorm + standard FFN, which doesn't match LLaMA's architecture.

## Decision

Build the LLaMA decoder layer manually from MLX primitives rather than using or subclassing `nn.TransformerDecoderLayer`.

The layer is constructed from:
- `nn.RMSNorm` — pre-attention and pre-FFN normalisation
- `nn.Linear` × 4 — Q, K, V projections (1024→1024) and output projection (1024→1024)
- `nn.RoPE(dims=64, base=500000.0)` — applied to Q and K after projection
- `mx.fast.scaled_dot_product_attention` — with causal mask and scale=1/√64
- `nn.Linear` × 3 — SwiGLU MLP (gate_proj + up_proj 1024→4096, down_proj 4096→1024)
- `nn.SiLU` — activation for SwiGLU gate

KV cache stored as `past_k, past_v` tuples per layer, passed explicitly in the autoregressive loop.

## Consequences

- **Positive**: Full control over RoPE application point. Matches PyTorch LLaMA exactly. No workarounds for API impedance mismatches.
- **Negative**: ~100 lines of layer code vs 1 line for `nn.TransformerDecoderLayer`. But this is boilerplate that any LLaMA port needs anyway.
- **Reversal cost**: Low — the layer is self-contained. If MLX later adds LLaMA support, replacing the manual layer is a swap.
