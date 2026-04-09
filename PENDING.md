# Pending Issues

## 1. torch.compile on multilingual T3 — blocked on upstream transformers bug (Step 8)

**Status:** Disabled — compile call commented out in `mtl_tts.py`

**Root cause:** transformers decorates `LlamaModel.forward()` with an `output_capturing.py`
wrapper that uses `torch` internally but never imports it at module level. When
`torch.compile` traces through it, this raises:
```
NameError: name 'torch' is not defined
  File "transformers/utils/output_capturing.py"
```
Confirmed broken in transformers 5.2.0 and 5.5.2. Opened upstream issue to track the fix.

**When re-enabling:** use `mode="default"` (not `"reduce-overhead"`), because
`AlignmentStreamAnalyzer`'s per-layer hooks are incompatible with CUDA graph replay.
The call is ready to uncomment in `mtl_tts.py` once the upstream bug is fixed.

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
