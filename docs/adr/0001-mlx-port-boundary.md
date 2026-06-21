# ADR-0001: MLX Port Boundary

**Status:** Accepted
**Date:** 2026-06-21

## Context

The T3 model's autoregressive speech token generation is the bottleneck (~15-22 tok/s on MPS vs the 25 tok/s needed for real-time). We need to accelerate it using MLX, Apple's Metal-native ML framework.

The T3 pipeline has four stages:
1. **Conditioning encoder** — VoiceEncoder + Perceiver resampler + speaker embedding projection (runs once per generation)
2. **Embedding tables** — text, speech, and position embeddings
3. **LLaMA backbone** — 30-layer decoder-only transformer (~95% of compute)
4. **Speech head** — linear projection to vocabulary logits

## Decision

Port stages 2, 3, and 4 (embeddings + backbone + speech head) to MLX. Stage 1 (conditioning encoder) remains in PyTorch.

The data flow across the boundary:
```
PyTorch → T3CondEnc → cond_emb (numpy)  
                    → text_tokens → text_emb (MLX)
                    → speech_tokens → speech_emb (MLX)
                    → [cond | text | speech] → LLaMA backbone (MLX) → speech_head (MLX) → logits
```

## Consequences

- **Positive**: The conditioning encoder is trivial complexity to keep in PyTorch (<50 lines of glue code). No MLX porting effort for the Perceiver or speaker encoder. The compute bottleneck (backbone) is fully on Metal.
- **Negative**: One numpy bridge copy between PyTorch and MLX per generation (negligible — 34×1024 float32 array). The conditioning encoder runs on PyTorch MPS in parallel with MLX device, but since it's a single forward pass this doesn't matter.
- **Reversal cost**: Low — the boundary is just a single array transfer point. Reversing (porting more to MLX) is equally easy.
