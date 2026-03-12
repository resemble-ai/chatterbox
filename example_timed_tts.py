# example_timed_tts.py
#
# Demonstrates timed narration with batching:
#   - Multiple scripts in one pass
#   - Multiple stochastic variants per script
#   - Per-group T3 passes → token resampling → single S3Gen pass

import os
import torch
import torchaudio as ta

from chatterbox.tts import ChatterboxTTS
from chatterbox.timed_tts import TimedChatterboxTTS, print_comfort_report

# ── Setup ────────────────────────────────────────────────────────────────────

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")
model = ChatterboxTTS.from_pretrained(device=device)
timed = TimedChatterboxTTS(model)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "tts_test_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

AUDIO_PROMPT = "C:/_myDrive/repos/auto-vlog/assets/audio_sample1.wav"   # ← replace with your voice reference


# # ════
# # PART 1 — Single-script generation (existing API, unchanged)
# # ════

# script = (
#     "The crow has landed on this building and started to systematically "
#     "dismantle the roof! <4.000> "
#     "Wow, look at that clever bird, the pieces are just flying! <7.000> "
#     "Yes, after a significant battle with the construction the feathered "
#     "rascal <9.000> "
#     "sets-off into the blue sky..."
# )

# print("=== Single script — expressive narrator ===")
# result = timed.generate(
#     script,
#     audio_prompt_path=AUDIO_PROMPT,
#     exaggeration=0.7,
#     cfg_weight=0.3,
#     temperature=0.8,
# )

# out_path = os.path.join(OUTPUT_DIR, "timed_narration.wav")
# ta.save(out_path, result.wav, result.sr)
# print(f"Saved {out_path}  ({result.total_duration:.2f}s)")
# print_comfort_report(result.segments)


# ════
# PART 2 — Batched generation: 2 scripts × 3 variants
# ════
#
# This runs ONE T3 call for all group texts across both scripts (with
# num_return_sequences=3 for variants), then ONE S3Gen call for all
# 2×3=6 final token streams.
#
# Benefits:
#   - GPU utilization: T3's KV-cache prefill is shared across variants
#   - S3Gen sees a padded batch → one kernel launch
#   - ~2-3× faster than calling generate() 6 times sequentially

texts_batch = [
    "This is the first sentence to be synthesized in a batch.",
    "This is the second one.",
    "And the dramatic third! Wow! It is so long that it lasts much, much longer than the preceding sentences! Check it for glitches.",
]

num_variants = 3

print(f"\n=== Batched generation: {len(texts_batch)} scripts × {num_variants} variants ===")

# Prepare conditioning once (shared across all scripts + variants)
model.prepare_conditionals(AUDIO_PROMPT, exaggeration=0.7)


batch_results = timed.generate_batch(
    texts_batch,
    exaggeration=[0.5, 0.7, 0.9],    # calm → expressive → very dramatic
    cfg_weight=[0.5, 0.3, 0.2],       # tight → loose → very loose guidance
    temperature=[0.7, 0.8, 0.9],      # conservative → creative sampling
    num_return_sequences=3,
)


# batch_results: List[List[TimedResult]]
#   batch_results[script_idx][variant_idx].wav  → (1, N) tensor
#   batch_results[script_idx][variant_idx].segments → comfort report

for s_idx, variant_list in enumerate(batch_results):
    for v_idx, result in enumerate(variant_list):
        fname = f"batch_script{s_idx+1}_variant{v_idx+1}.wav"
        fpath = os.path.join(OUTPUT_DIR, fname)
        ta.save(fpath, result.wav, result.sr)
        print(f"\n  Script {s_idx+1}, Variant {v_idx+1}: {fpath}  ({result.total_duration:.2f}s)")
        print_comfort_report(result.segments)

print("\nComfort scores per variant:")
for s_idx, variant_list in enumerate(batch_results):
    for v_idx, result in enumerate(variant_list):
        scores = [seg.comfort for seg in result.segments]
        print(f"  Script {s_idx+1} v{v_idx+1}: {scores}")


# # ════
# # PART 3 — Batched with silence gaps
# # ════

# scripts_with_silence = [
#     "And here we go! <2.000> <4.000> Back to action! <6.000> The end.",
#     "A quiet <1.500> <3.000> then sudden excitement! <4.500> Wow!",
# ]

# print(f"\n=== Batched with silence gaps: {len(scripts_with_silence)} scripts × 2 variants ===")

# batch_results2 = timed.generate_batch(
#     scripts_with_silence,
#     exaggeration=0.7,
#     cfg_weight=0.3,
#     num_return_sequences=2,
# )

# for s_idx, variant_list in enumerate(batch_results2):
#     for v_idx, result in enumerate(variant_list):
#         fname = f"batch_silence_s{s_idx+1}_v{v_idx+1}.wav"
#         fpath = os.path.join(OUTPUT_DIR, fname)
#         ta.save(fpath, result.wav, result.sr)
#         print(f"\n  Script {s_idx+1}, Variant {v_idx+1}: {fpath}")
#         print_comfort_report(result.segments)


# ════
# Interpreting comfort / Parameter guide 
# ════
#
#  0.50       = perfect — no resampling needed
#  0.35–0.49  = mild token repetition (slightly slower)
#  0.51–0.65  = mild token dropping (slightly faster)
#  < 0.15     = extreme — tell Gemini to add more words
#  > 0.85     = extreme — tell Gemini to shorten text or extend window
#
#  exaggeration  cfg_weight  Style
#  ───────────── ─────────── ─────────────────────────────────────────
#  0.5           0.5         Neutral / default — good for informational VO
#  0.7           0.3         Expressive narrator — pitch peaks, emphasis
#  0.8–1.0       0.2–0.3    Very dramatic — comedy, horror, sports
#  0.3           0.5         Subdued / calm — documentary, meditation