# example_timed_tts.py
#
# Demonstrates timed narration: single T3 pass → token resampling → single S3Gen pass.
# Each segment's duration is controlled at the token level (1 token = 40ms).

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

# ── The narrator script ──────────────────────────────────────────────────────

script = (
    "The crow has landed on this building and started to systematically "
    "dismantle the roof! <4.000> "
    "Wow, look at that clever bird, the pieces are just flying! <7.000> "
    "Yes, after a significant battle with the construction the feathered "
    "rascal <9.000> "
    "sets-off into the blue sky..."
)

# ── Step 1: Instant comfort estimation ───────────────────────────────────────

print("=== Comfort Estimation (no GPU) ===\n")
estimates = timed.estimate_comfort(script)
for i, est in enumerate(estimates):
    tgt = f"{est['target_duration']:.2f}s" if est['target_duration'] is not None else "free"
    print(
        f"  Seg {i}: comfort={est['estimated_comfort']:.2f}  "
        f"natural≈{est['estimated_natural_duration']:.1f}s  target={tgt}  "
        f'"{est["text"][:55]}…"'
    )
print()

# ── Step 2: Full generation ──────────────────────────────────────────────────

print("=== Generating (single T3 pass → token resample → single S3Gen pass) ===")
result = timed.generate(
    script,
    audio_prompt_path=AUDIO_PROMPT,
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8,
)

out_path = os.path.join(OUTPUT_DIR, "timed_narration.wav")
ta.save(out_path, result.wav, result.sr)
print(f"Saved {out_path}  ({result.total_duration:.2f}s)")
print_comfort_report(result.segments)

print("Comfort scores:", [seg.comfort for seg in result.segments])
print()

# ── Example with silence gaps ────────────────────────────────────────────────

script_silence = (
    "And here we go! <2.000> "
    "<4.000> "                         # 2s silence
    "Back to action! <6.000> "
    "The end."
)

print("=== Script with silence gaps ===")
result2 = timed.generate(script_silence)
out2 = os.path.join(OUTPUT_DIR, "timed_with_silence.wav")
ta.save(out2, result2.wav, result2.sr)
print(f"Saved {out2}")
print_comfort_report(result2.segments)

# ── Interpreting comfort ─────────────────────────────────────────────────────
#
#  0.50       = perfect — no resampling needed
#  0.35–0.49  = mild token repetition (slightly slower)
#  0.51–0.65  = mild token dropping (slightly faster)
#  < 0.15     = extreme — tell Gemini to add more words
#  > 0.85     = extreme — tell Gemini to shorten text or extend window