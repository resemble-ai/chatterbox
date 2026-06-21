# ADR-0005: Success Criteria

**Status:** Accepted  
**Date:** 2026-06-21

## Context

Without a clear definition of "success," the PoC has no termination condition.

## Decision

| Metric | Threshold | Method |
|---|---|---|
| **Generation speed** | ≥25 tok/s (1× real-time) | Average over 3 runs of 200-token generation, warm cache |
| **Output correctness** | Logit MSE < 1e-2 vs PyTorch baseline | Single forward pass, same random seed |
| **Drop-in compatibility** | MLX-generated speech tokens → PyTorch S3Gen produces audible speech | Subjective: no obvious artifacts |

## Consequences

- **Positive**: Clear go/no-go for the MLX approach. If MLX achieves ≥25 tok/s, the approach is viable; if <15 tok/s, it's not worth the porting complexity.
- **Negative**: None — objective criteria.
