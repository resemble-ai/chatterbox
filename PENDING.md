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

## 2. Dockerfile: gcc/g++ added for torch.compile

**Status:** Done (in Dockerfile), but only needed once torch.compile is re-enabled.

The `gcc` and `g++` packages were added to the Dockerfile to support `torch.compile`'s
Triton/inductor backend. These can be removed if torch.compile is not used, but are
harmless to keep (~50MB).
