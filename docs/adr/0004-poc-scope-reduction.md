# ADR-0004: PoC Scope Reduction

**Status:** Accepted  
**Date:** 2026-06-21

## Context

The T3 model includes several peripheral components that contribute minimal compute cost. For a proof-of-concept focused on backbone speed, porting these is pure overhead.

## Decision

Port the minimal subset needed for a meaningful speed benchmark:

| Component | Decision | Rationale |
|---|---|---|
| **Perceiver resampler** | Skip | Runs once per generation, not per token. Keep in PyTorch, pass cond_emb as numpy array. |
| **CFG (classifier-free guidance)** | Skip | `cfg_weight=0` for PoC. Standard T3 `inference_turbo` already skips it. |
| **Emotion conditioning** | Skip | `emotion_adv` is a scalar → 1024 linear. Trivial to add later if needed. |
| **CLAP embedding** | Skip | Always None in current code. Dead code path. |
| **Text logit head** | Skip | Only needed for training. PoC is inference-only. |
| **Top-p / min-p sampling** | Include | Needed for quality output. ~20 lines. |
| **Repetition penalty** | Include | Needed for quality output. ~5 lines. |
| **KV cache** | Include | Critical for performance. Without it, attention cost is O(n²) per step. |

## Consequences

- **Positive**: PoC code is focused on the bottleneck. ~350 lines instead of ~600+. Faster time to benchmark.
- **Negative**: Can't test full end-to-end audio without PyTorch glue. PoC benchmark measures tokens/sec, not audio output.
- **Reversal cost**: Low — each skipped component is self-contained. Adding them later doesn't require refactoring.
