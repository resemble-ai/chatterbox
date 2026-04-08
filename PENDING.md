# Pending Issues

## 1. torch.compile crashes on base and turbo models (Step 6)

**Status:** Disabled — compile calls commented out in `tts.py:165` and `tts_turbo.py:179`

**Problem:** `torch.compile(mode="reduce-overhead")` uses CUDA graphs, which require
static tensor shapes. Both base (Llama) and turbo (GPT2) use a dynamic KV cache that
grows via `torch.cat` each iteration, which is incompatible with CUDA graphs.

**Errors:**
- **Turbo (GPT2):** `RuntimeError: accessing tensor output of CUDAGraphs that has been
  overwritten by a subsequent run` — the dynamic `DynamicCache.update()` in
  `transformers/cache_utils.py:120` calls `torch.cat([self.keys, key_states], dim=-2)`,
  which mutates tensor shapes between graph replays.
- **Base (Llama):** `AssertionError` in `torch/_inductor/cudagraph_trees.py:2411` —
  `len(node.tensor_weakrefs) == len(node.stack_traces)` fails during warmup because
  the dynamic cache creates new tensors each step.

**Fix path (corresponds to plan Step 9):**
1. Replace `DynamicCache` with `StaticCache` (HuggingFace `transformers.StaticCache`)
   - Pre-allocate KV tensors to `max_gen_len` so shapes are fixed
   - Pass `cache_position` tensor to track current position
2. After StaticCache works, re-enable `t3.compile_for_inference()` in both files
3. Consider `torch.compiler.cudagraph_mark_step_begin()` before each forward call
   as an alternative if StaticCache proves difficult

**Also noted:** Warning about TensorFloat32 — can be addressed by adding
`torch.set_float32_matmul_precision('high')` in model init for a free speedup on
Ampere+ GPUs (RTX 30xx, A100, etc).

**Files to modify:**
- `src/chatterbox/models/t3/t3.py` — inference loops need StaticCache + cache_position
- `src/chatterbox/tts.py` — re-enable compile call
- `src/chatterbox/tts_turbo.py` — re-enable compile call
- `src/chatterbox/models/t3/inference/t3_hf_backend.py` — forward() may need cache handling

**References:**
- HuggingFace StaticCache docs: search for `transformers.StaticCache`
- `compile_for_inference()` method is in `t3.py:93`
- Plan: see step 9 in the optimization plan

## 2. torch.compile on multilingual T3 (Step 8)

**Status:** Disabled — compile call commented out in `mtl_tts.py:201`

**Problem:** `torch.compile(mode="default")` hits a `NameError: name 'torch' is not
defined` inside `transformers/utils/output_capturing.py`. This is a compatibility issue
between the installed transformers version and torch.compile's dynamo tracer.

**Fix path:** Upgrade transformers to a version with torch.compile support, or pin a
compatible pair of torch + transformers versions. Then re-enable the commented-out call
in `mtl_tts.py`.

## 3. torch.compile on S3Gen vocoder (Step 10)

**Status:** Disabled — compile call commented out in `mtl_tts.py:207`

**Problem:** The S3Gen estimator has too many graph breaks (`.item()` calls in
`models/s3gen/utils/mask.py:161`, dynamic tensor shapes per chunk) causing excessive
recompilation. First inference was 5x slower; steady-state was also slower than
uncompiled.

**Fix path:** Refactor the `.item()` calls and data-dependent control flow in the
estimator's mask computation to be torch.compile-friendly, or use
`torch._dynamo.config.capture_scalar_outputs = True`.

## 4. Dockerfile: gcc/g++ added for torch.compile

**Status:** Done (in Dockerfile), but only needed once torch.compile is re-enabled.

The `gcc` and `g++` packages were added to the Dockerfile to support `torch.compile`'s
Triton/inductor backend. These can be removed if torch.compile is not used, but are
harmless to keep (~50MB).
