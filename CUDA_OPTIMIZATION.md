# T3 Autoregressive Loop — CUDA Optimization Brief

## Problem

The server currently runs at RTF ≈ 1.2–1.3 (it takes 1.2–1.3 s to synthesize 1 s of audio).
GPU utilization sits at 50–70% during synthesis. The GPU is idle roughly a third of the time —
not because it lacks work, but because the Python interpreter is preparing the next step.

The bottleneck is the **autoregressive T3 token generation loop**. Each iteration does:

1. Python prepares the next input embedding
2. CPU → GPU transfer of a single-token tensor
3. GPU runs one transformer forward pass
4. GPU → CPU sync to read the output token (to check for EOS)
5. Python applies logit processors (CFG, temperature, rep penalty, min-p, top-p)
6. Python samples the next token
7. Repeat up to ~1000 times

Steps 4 and 6 both force a GPU→CPU synchronization every single token. At 25 tokens/sec of
audio, a 10-second output requires ~250 synchronizations. Each sync costs ~0.1–0.5 ms of idle
time on top of the actual compute, which adds up to significant wall-clock overhead at small
batch sizes.

---

## Where the loop lives

There are three implementations, one per model variant. They are structurally identical:

| File | Class | Method |
|------|-------|--------|
| `src/chatterbox/tts.py` | `ChatterboxTTS` | `_inference_stream` (line ~286) |
| `src/chatterbox/mtl_tts.py` | `ChatterboxMultilingualTTS` | `_inference_stream` |
| `src/chatterbox/tts_turbo.py` | `ChatterboxTurboTTS` | `_inference_stream_turbo` |

The `base` and `multilingual` loops are nearly identical. `turbo` uses a GPT2 backbone instead
of Llama and has no CFG, so it is simpler.

### The loop in detail (`tts.py:_inference_stream`, simplified)

```python
# --- ONE-TIME SETUP (before the loop) ---

# Build the patched HF model wrapper
patched_model = T3HuggingfaceBackend(
    config=self.t3.cfg,
    llama=self.t3.tfmr,          # the actual LlamaModel
    speech_enc=self.t3.speech_emb,
    speech_head=self.t3.speech_head,
    alignment_stream_analyzer=alignment_stream_analyzer,  # multilingual only
)

# Prefill: run the full conditioning + text prefix through the model once
# inputs_embeds shape: (2, prefix_len, dim)  — batch=2 because of CFG
output = patched_model(inputs_embeds=inputs_embeds, past_key_values=None, use_cache=True, ...)
past = output.past_key_values   # KV cache, grows by 1 each step

# --- THE HOT LOOP ---
for i in range(max_new_tokens):          # up to ~1000 iterations
    logits = output.logits[:, -1, :]     # (2, vocab_size)

    # CFG: combine conditional and unconditional logits
    logits = logits[0:1] + cfg_weight * (logits[0:1] - logits[1:2])

    # Logit processors (all Python, all on GPU tensors but with potential syncs)
    logits = rep_pen(generated_ids, logits)
    logits = logits / temperature
    logits = min_p_warper(generated_ids, logits)
    logits = top_p_warper(generated_ids, logits)
    probs  = torch.softmax(logits, dim=-1)

    next_token = torch.multinomial(probs, num_samples=1)   # GPU op

    # ← SYNC POINT: .view(-1) == stop_token forces GPU→CPU read
    if next_token.view(-1) == self.t3.hp.stop_speech_token:
        break

    # Append to buffer, yield chunk every `chunk_size` tokens
    chunk_buffer.append(next_token)
    generated_ids = torch.cat([generated_ids, next_token], dim=1)

    # Build next input embedding (positional + token embedding — tiny GPU op)
    next_token_embed = self.t3.speech_emb(next_token)
    next_token_embed = next_token_embed + self.t3.speech_pos_emb.get_fixed_embedding(i + 1)
    next_token_embed = torch.cat([next_token_embed, next_token_embed])  # CFG duplicate

    # ← ONE TRANSFORMER FORWARD PASS per token (the expensive GPU op)
    output = patched_model(
        inputs_embeds=next_token_embed,   # shape: (2, 1, dim)
        past_key_values=past,
        use_cache=True,
        ...
    )
    past = output.past_key_values
```

### Key properties of the loop

- **Batch size is always 2** for `base`/`multilingual` (CFG requires one conditional + one
  unconditional pass). `turbo` has batch size 1.
- **KV cache grows dynamically** — `past_key_values` adds one new (key, value) slice per
  step across all attention layers.
- **The transformer used is `LlamaModel`** (HuggingFace implementation) for `base` and
  `multilingual`, `GPT2Model` for `turbo`. Accessed via `T3HuggingfaceBackend` which wraps
  it with custom embedding and logit projection layers.
- **`output_attentions=True`** is passed on every step for the multilingual model because
  `AlignmentStreamAnalyzer` reads attention maps to detect repetition/hallucinations. This
  disables FlashAttention for the multilingual model — see the section on that below.
- **The EOS check** (`next_token.view(-1) == stop_token`) is a Python comparison on a GPU
  tensor — PyTorch implicitly calls `.item()` here, synchronizing the GPU every step.

---

## The T3HuggingfaceBackend wrapper

`src/chatterbox/models/t3/inference/t3_hf_backend.py`

This is a thin `LlamaPreTrainedModel + GenerationMixin` subclass. In inference it is used
directly (not via HF `generate()`). Its `forward()` does:

1. `LlamaModel.forward(inputs_embeds=..., past_key_values=..., use_cache=True)`
2. Extract last hidden state
3. `speech_head(hidden_states)` → logits

The `alignment_stream_analyzer` hook is currently commented out in `forward()` (line ~109)
and is instead called manually in the loop in `_inference_stream`. This matters for
`torch.compile` because the hook reads `.attentions` from the output, which forces
`output_attentions=True`, which breaks FlashAttention.

---

## The AlignmentStreamAnalyzer (multilingual only)

`src/chatterbox/models/t3/inference/alignment_stream_analyzer.py`

This module injects a forward hook into one specific attention layer (index 9 by default)
to read attention weights and detect hallucinations/repetitions. It runs on every token step
in the multilingual model.

The hook uses `output_attentions=True` which forces PyTorch/HF to materialize all attention
weight tensors rather than using memory-efficient kernels (FlashAttention, SDPA). For the
multilingual Llama model, this is a **significant overhead on its own**, independent of the
Python loop issue.

---

## Recommended optimization approach

### 1. `torch.compile` on the transformer forward pass (highest value, lowest risk)

Wrap the `LlamaModel` (or `T3HuggingfaceBackend`) forward call with `torch.compile`. This
fuses the transformer's internal operations — attention, FFN, layer norms, residuals — into
optimized CUDA kernels per step. It does **not** eliminate the Python loop or the GPU→CPU
sync, but it reduces the per-step GPU compute time, which indirectly reduces idle time.

```python
# In T3HuggingfaceBackend.__init__ or just before the first inference call:
import torch
patched_model.model = torch.compile(patched_model.model, mode="reduce-overhead")
```

`mode="reduce-overhead"` is specifically designed for loops with small, repeated workloads
(exactly this pattern). `mode="max-autotune"` gives more gains but has a longer warmup.

**Caveats:**
- First call will be slow (compilation). The server's `--warmup` flag handles this.
- Dynamic KV cache shapes (grows each step) will trigger recompilation unless the KV cache
  is padded to a fixed maximum length.
- `output_attentions=True` (required by `AlignmentStreamAnalyzer`) is incompatible with
  `torch.compile` + `torch.nn.attention.sdpa_kernel`. This needs to be resolved first for
  the multilingual model (see item 3 below).

### 2. CUDA graphs (highest potential gain, highest complexity)

CUDA graphs capture the entire GPU work for one loop iteration into a single replayable
graph, eliminating Python overhead between steps entirely. GPU utilization would approach
100% during the loop.

The main obstacle is that CUDA graphs require **static tensor shapes**. The KV cache grows
by one slice per step, making shapes dynamic. The standard workaround is to pre-allocate a
KV cache tensor of maximum length and track a position index, rather than appending tensors.

This requires either:
- Implementing a static KV cache (similar to what `torch.export` + `executorch` does for
  LLaMA), or
- Using HuggingFace's `StaticCache` (available since transformers ≥ 4.38) which is designed
  exactly for this and is compatible with `torch.compile` + CUDA graphs:

```python
from transformers import StaticCache

past_key_values = StaticCache(
    config=patched_model.config,
    max_batch_size=2,           # CFG doubles batch
    max_cache_len=1200,         # max tokens (prefix + generated)
    device=device,
    dtype=model_dtype,
)
```

Then pass this cache into the loop instead of growing `past_key_values` dynamically. With
static cache shapes, `torch.compile(model, mode="reduce-overhead")` can use CUDA graphs
automatically when `torch._inductor.config.triton.cudagraph_trees = True` (default in recent
PyTorch).

### 3. Fix `output_attentions=True` for multilingual (prerequisite for compile)

`AlignmentStreamAnalyzer` uses a forward hook on attention layer 9. It currently forces
`output_attentions=True` for the entire model, which:
- Disables FlashAttention / SDPA for **all** layers
- Is incompatible with `torch.compile` in its current form

The fix is to register a `register_forward_hook` directly on layer 9's attention module
instead of relying on `output_attentions=True`. The hook captures the attention weights for
that layer only, without affecting the rest. This restores FlashAttention for all other
layers and makes `torch.compile` viable for multilingual.

The relevant attention module path in the multilingual LlamaModel:
```python
model.tfmr.layers[9].self_attn
```

Register a hook that stores the attention output:
```python
captured = {}
def _attn_hook(module, input, output):
    # output is (hidden_state, attn_weights, past_kv) for output_attentions=True
    # with a hook we can intercept the raw attn weights if the layer supports it
    captured['attn'] = output[1] if isinstance(output, tuple) else None

model.tfmr.layers[9].self_attn.register_forward_hook(_attn_hook)
```

Then pass `captured` to `AlignmentStreamAnalyzer` instead of relying on the output dict.

### 4. EOS check without GPU→CPU sync

Replace the Python comparison:
```python
if next_token.view(-1) == self.t3.hp.stop_speech_token:
    break
```

with a lazy check that only syncs every N steps (or defers to the chunk boundary):
```python
# Check EOS only at chunk boundaries — worst case adds chunk_size - 1 extra tokens
if len(chunk_buffer) >= chunk_size or (i % 8 == 0 and next_token.item() == stop_token):
    ...
```

Or buffer the EOS check using `torch.any` on the accumulated `generated_ids` tensor without
calling `.item()`, and only break at the next yield point.

---

## Files to modify (summary)

| File | What to change |
|------|----------------|
| `src/chatterbox/models/t3/inference/t3_hf_backend.py` | Apply `torch.compile` to `self.model`; optionally switch to `StaticCache` |
| `src/chatterbox/models/t3/inference/alignment_stream_analyzer.py` | Replace `output_attentions=True` with a layer-specific forward hook |
| `src/chatterbox/tts.py` | `_inference_stream`: replace dynamic KV cache with `StaticCache`; defer EOS sync |
| `src/chatterbox/mtl_tts.py` | Same as `tts.py` |
| `src/chatterbox/tts_turbo.py` | `_inference_stream_turbo`: simpler (no CFG, GPT2 backbone, no AlignmentStreamAnalyzer) — good candidate for a first `torch.compile` proof of concept |
| `server/server.py` | The `--warmup` flag already handles compile warmup; may need to increase warmup text length |

---

## Suggested implementation order

1. **Start with `turbo`** — no CFG, no AlignmentStreamAnalyzer, GPT2 backbone. Apply
   `torch.compile(mode="reduce-overhead")` to `patched_model.model` and measure RTF delta.
   This is the lowest-risk proof of concept.

2. **Fix `output_attentions` hook** for multilingual — unblock FlashAttention and
   `torch.compile` for the main model.

3. **Apply `torch.compile`** to `base`/`multilingual` after step 2.

4. **Add `StaticCache`** if compile alone doesn't reach target RTF — this enables CUDA
   graphs and eliminates the remaining Python overhead.

5. **Defer EOS sync** as a small complementary win once compile is in place.

---

## Environment

- Python 3.11, PyTorch (conda), CUDA
- HuggingFace `transformers` (version determines `StaticCache` availability — check
  `transformers.__version__ >= "4.38"`)
- Server: `server/server.py` (FastAPI + uvicorn)
- `--warmup` CLI flag pre-runs one synthesis on startup to trigger JIT compilation

## Current RTF baseline

| Condition | RTF |
|-----------|-----|
| fp32, no compile (current) | ~1.2–1.3 |
| Target for real-time streaming | < 0.8 |
