# Pending Issues

## 1. torch.compile on multilingual T3 — verify in production (Step 8)

**Status:** Re-enabled with `mode="default"`. Bumped transformers 5.2.0 → 5.5.2 to pick up
the `output_capturing.py` fix (missing `import torch` caused NameError on compile).

**What changed:** `mtl_tts.py` calls `t3.compile_for_inference(mode="default")` on CUDA.
`mode="default"` uses inductor kernel fusion without CUDA graphs. CUDA graphs are skipped
because `AlignmentStreamAnalyzer` runs Python-side hooks between generation steps, which is
incompatible with CUDA graph replay.

**Still to verify:** confirm RTF improvement in a real deployment run and that audio quality
is unchanged. If the NameError recurs (5.5.2 still broken), fall back to patching the file
in the Dockerfile by prepending `import torch\n` to `output_capturing.py` after pip install.

---

## 2. torch.compile on S3Gen vocoder (Step 10)

**Status:** Still disabled — compile call commented out in `mtl_tts.py`

**What improved:** The `.item()` call in `models/s3gen/utils/mask.py` that caused a graph
break has been replaced with a pure tensor op (no `.item()`). This removes one graph break.

**Remaining blockers:**
- Dynamic tensor shapes per streaming chunk (different chunk sizes) still cause
  recompilation on every new chunk size. This dominates the overhead.
- Steady-state RTF was worse compiled than uncompiled in initial testing.

**Fix path:** Pad speech token chunks to a fixed size (e.g. always `chunk_size` tokens,
zero-padded) before passing to S3Gen, so shapes are static. Only attempt after the T3
improvements are confirmed stable.

---

## 3. StaticCache + torch.compile for non-streaming `t3.inference()` (Step 9 partial)

**Status:** Not yet done

The streaming paths (`_inference_stream` in `tts.py` and `mtl_tts.py`) now use `StaticCache`.
The non-streaming `t3.inference()` method in `t3.py` and the turbo `inference_turbo` /
`inference_turbo_stream` methods still use the legacy dynamic KV cache. These are lower
priority since the server always uses the streaming path for real-time generation.

**Fix path:**
- `t3.inference()`: add `StaticCache` + `cache_position` (same pattern as `_inference_stream`)
- `inference_turbo` / `inference_turbo_stream`: GPT2 may need a different cache approach;
  test whether `mode="default"` compile alone is sufficient without StaticCache for GPT2
