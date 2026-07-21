# ADR-0003: Weight Conversion Strategy

**Status:** Accepted  
**Date:** 2026-06-21

## Context

The T3 model weights are distributed as HuggingFace safetensors files (`.safetensors`). MLX requires its own array format. We need a conversion strategy that works for both offline (saved `.npz` checkpoint) and online (direct load) use cases.

## Decision

Load safetensors with PyTorch, then convert each tensor to `mx.array` and build the MLX model's state dict by key-name mapping.

Key mapping convention (PyTorch → MLX):
```
tfmr.layers.{n}.self_attn.q_proj.weight → model.layers.{n}.attention.q_proj.weight
tfmr.layers.{n}.self_attn.k_proj.weight → model.layers.{n}.attention.k_proj.weight
tfmr.layers.{n}.self_attn.v_proj.weight → model.layers.{n}.attention.v_proj.weight
tfmr.layers.{n}.self_attn.o_proj.weight → model.layers.{n}.attention.o_proj.weight
tfmr.layers.{n}.input_layernorm.weight  → model.layers.{n}.attention_norm.weight
tfmr.layers.{n}.post_attention_layernorm.weight → model.layers.{n}.ffn_norm.weight
tfmr.layers.{n}.mlp.gate_proj.weight    → model.layers.{n}.mlp.gate.weight
tfmr.layers.{n}.mlp.up_proj.weight      → model.layers.{n}.mlp.up.weight
tfmr.layers.{n}.mlp.down_proj.weight    → model.layers.{n}.mlp.down.weight
tfmr.norm.weight                         → model.norm.weight
speech_emb.weight                        → model.speech_emb.weight
speech_head.weight                       → model.speech_head.weight
text_emb.weight                          → model.text_emb.weight
text_head.weight                         → model.text_head.weight
speech_pos_emb.emb.weight                → model.speech_pos_emb.weight
text_pos_emb.emb.weight                  → model.text_pos_emb.weight
```

Non-ported weights (remain in PyTorch, not loaded into MLX):
- `cond_enc.*` — stays in PyTorch
- `tfmr.embed_tokens.weight` — unused in T3 (model passes `inputs_embeds`, not token IDs)

## Consequences

- **Positive**: Direct key-name mapping is simple, auditable, and reversible. No custom serialization format. Works with any safetensors checkpoint.
- **Negative**: Requires both PyTorch and MLX installed. Adds ~100ms load time for the PyTorch→MLX conversion. Acceptable for a PoC.
- **Reversal cost**: Low — the conversion is a one-way script. Switching to pure MLX loading (numpy `.npz` format) would be a day's work.
