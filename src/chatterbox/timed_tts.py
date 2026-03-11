# src/chatterbox/timed_tts.py
#
# Timed narration for Chatterbox TTS.
#
# Generates a SINGLE T3 pass for natural prosody across the full text,
# then resamples the speech tokens per segment to hit target durations,
# then runs a SINGLE S3Gen pass for a coherent waveform.
#
# Token resampling (nearest-neighbour on discrete speech tokens) is far
# cleaner than phase-vocoder stretching on the final audio:
#   - No metallic/tube artifacts
#   - Duration control is exact (1 token = 40 ms)
#   - S3Gen still sees one continuous token stream → one coherent voice
#
# For ±20% adjustments the quality is transparent.  Larger adjustments
# (flagged by the comfort score) should be addressed by asking Gemini
# to adjust the text length rather than forcing extreme resampling.
#
# Input format:
#   "The crow landed! <4.000> Wow! <7.000> Off it goes..."
#
# Usage:
#   from chatterbox.tts import ChatterboxTTS
#   from chatterbox.timed_tts import TimedChatterboxTTS
#
#   model = ChatterboxTTS.from_pretrained(device="cuda")
#   timed = TimedChatterboxTTS(model)
#   result = timed.generate("Hello! <2.0> World!", audio_prompt_path="ref.wav")
#   torchaudio.save("out.wav", result.wav, result.sr)

import math
import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from .tts import ChatterboxTTS, punc_norm
from .models.s3gen.const import S3GEN_SR, TOKEN_TO_WAV_RATIO

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

SECS_PER_TOKEN = TOKEN_TO_WAV_RATIO / S3GEN_SR    # 0.04 s
SAMPLES_PER_TOKEN = TOKEN_TO_WAV_RATIO             # 960

# Comfort estimation heuristic: average speech tokens per text token.
_AVG_SPEECH_TOKENS_PER_TEXT_TOKEN = 2.1

# Fade duration (ms) at silence-gap boundaries in the final waveform.
_SILENCE_FADE_MS = 60


# ── Timing-tag regex ─────────────────────────────────────────────────────────

_TAG_RE = re.compile(r"<(\d+(?:\.\d+)?)>")


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class TimedSegment:
    """One chunk of narration text with its target time window."""
    text: str
    start_time: float
    end_time: Optional[float]

    @property
    def target_duration(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def is_silence(self) -> bool:
        return len(self.text.strip()) == 0


@dataclass
class SegmentResult:
    """Per-segment diagnostics."""
    text: str
    start_time: float
    end_time: float
    target_duration: Optional[float]
    natural_duration: float
    actual_duration: float
    tokens_original: int
    tokens_resampled: int
    comfort: float              # 0.0 = too slow … 0.5 = natural … 1.0 = garbled


@dataclass
class TimedResult:
    """Return value of generate()."""
    wav: torch.Tensor           # (1, num_samples) at 24 kHz
    sr: int
    segments: List[SegmentResult]
    total_duration: float


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_timed_text(raw: str) -> List[TimedSegment]:
    """
    Split text on ``<float>`` timing tags.

    Empty text between adjacent tags becomes a silence gap.
    The last segment (after the final tag) is open-ended.
    """
    parts = _TAG_RE.split(raw)
    segments: List[TimedSegment] = []
    cursor = 0.0

    for i in range(0, len(parts), 2):
        seg_text = parts[i].strip()
        end_time = float(parts[i + 1]) if i + 1 < len(parts) else None

        if end_time is not None and end_time < cursor:
            logger.warning(
                "Timestamp %.3f < cursor %.3f — clamping.", end_time, cursor
            )
            end_time = cursor + 0.01

        segments.append(TimedSegment(text=seg_text, start_time=cursor, end_time=end_time))
        if end_time is not None:
            cursor = end_time

    return segments


# ── Comfort score ────────────────────────────────────────────────────────────

def _comfort_score(natural_dur: float, target_dur: float,
                   steepness: float = 3.5) -> float:
    """
    0.5 = natural pace.  → 0 = too slow.  → 1 = too fast.

    Uses a logistic curve: comfort = 1 / (1 + exp(steepness × (ratio − 1)))
    where ratio = target_dur / natural_dur.
    """
    if natural_dur <= 0 or target_dur <= 0:
        return 0.5
    ratio = target_dur / natural_dur
    return 1.0 / (1.0 + math.exp(steepness * (ratio - 1.0)))


# ── Token resampling ─────────────────────────────────────────────────────────

def _resample_tokens(tokens: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Resample a 1-D tensor of discrete speech tokens to ``target_len``
    using nearest-neighbour indexing.

    - target_len > len(tokens) → tokens are repeated (slower speech)
    - target_len < len(tokens) → tokens are dropped  (faster speech)

    For adjustments within ±20% the S3Gen conformer handles this
    transparently.  Larger ratios should be avoided.
    """
    n = len(tokens)
    if n == 0 or target_len <= 0:
        return tokens[:0]
    if target_len == n:
        return tokens
    indices = torch.linspace(0, n - 1, target_len).round().long()
    return tokens[indices]


# ── Fade helpers (for silence-gap boundaries in final waveform) ──────────────

def _apply_fade_out(audio_np: np.ndarray, sr: int) -> np.ndarray:
    fade = min(int(sr * _SILENCE_FADE_MS / 1000), len(audio_np))
    if fade <= 1:
        return audio_np
    out = audio_np.copy()
    t = np.linspace(0, np.pi / 2, fade, dtype=np.float32)
    out[-fade:] *= np.cos(t) ** 2
    return out


def _apply_fade_in(audio_np: np.ndarray, sr: int) -> np.ndarray:
    fade = min(int(sr * _SILENCE_FADE_MS / 1000), len(audio_np))
    if fade <= 1:
        return audio_np
    out = audio_np.copy()
    t = np.linspace(0, np.pi / 2, fade, dtype=np.float32)
    out[:fade] *= np.sin(t) ** 2
    return out


# ── Main class ───────────────────────────────────────────────────────────────

class TimedChatterboxTTS:
    """
    Timing-aware wrapper around :class:`ChatterboxTTS`.

    Pipeline:
      1. Parse ``<timestamp>`` tags → segments with target durations.
      2. **Single T3 pass** on the full stripped text → natural speech tokens.
      3. Estimate per-segment token boundaries (proportional to text-token count).
      4. **Resample** each segment's tokens to hit its target duration
         (1 token = 40 ms, so target_tokens = round(target_dur / 0.04)).
      5. Concatenate resampled tokens (grouping around silence gaps).
      6. **S3Gen pass(es)** → coherent waveform(s).
      7. Assemble final audio with absolute positioning + silence gaps.
    """

    def __init__(self, model: ChatterboxTTS):
        self.model = model
        self.sr = model.sr

    # ── Cheap pre-check ──────────────────────────────────────────────────

    def estimate_comfort(self, timed_text: str) -> List[dict]:
        """Instantly estimate comfort scores without running TTS."""
        segments = parse_timed_text(timed_text)
        results = []
        for seg in segments:
            if seg.is_silence:
                results.append(dict(
                    text="", target_duration=seg.target_duration,
                    estimated_natural_duration=0.0, estimated_comfort=0.5,
                ))
                continue
            normed = punc_norm(seg.text)
            n_text = len(self.model.tokenizer.encode(normed))
            est_natural = n_text * _AVG_SPEECH_TOKENS_PER_TEXT_TOKEN * SECS_PER_TOKEN
            if seg.target_duration and seg.target_duration > 0:
                comfort = _comfort_score(est_natural, seg.target_duration)
            else:
                comfort = 0.5
            results.append(dict(
                text=seg.text, target_duration=seg.target_duration,
                estimated_natural_duration=round(est_natural, 3),
                estimated_comfort=round(comfort, 3),
            ))
        return results

    # ── Full generation ──────────────────────────────────────────────────

    def generate(
        self,
        timed_text: str,
        audio_prompt_path: Union[str, List[str]] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        top_p: float = 1.0,
        *,
        comfort_steepness: float = 3.5,
    ) -> TimedResult:
        """
        Generate time-aligned narration.

        Single T3 pass → token resampling → single S3Gen pass.
        Silence gaps are handled at the waveform level.
        """

        # ── 1. Parse ─────────────────────────────────────────────────────
        segments = parse_timed_text(timed_text)
        if not segments:
            raise ValueError("No segments found.")

        speech_segments = [s for s in segments if not s.is_silence]
        if not speech_segments:
            raise ValueError("All segments empty.")

        # ── 2. Single T3 pass on the full text ──────────────────────────
        full_text = " ".join(seg.text for seg in speech_segments)
        logger.info("T3 generating for: %.120s…", full_text)

        if audio_prompt_path:
            self.model.prepare_conditionals(
                audio_prompt_path, exaggeration=exaggeration
            )

        all_tokens = self.model.generate_speech_tokens(
            full_text,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        )
        # all_tokens: 1-D LongTensor on CPU
        total_tokens = len(all_tokens)
        logger.info(
            "T3 produced %d tokens (%.2fs at natural pace).",
            total_tokens, total_tokens * SECS_PER_TOKEN,
        )

        # ── 3. Estimate per-segment token boundaries ────────────────────
        #
        # Proportional to text-token counts.
        text_token_counts = []
        for seg in speech_segments:
            normed = punc_norm(seg.text)
            toks = self.model.tokenizer.text_to_tokens(normed).squeeze(0)
            text_token_counts.append(max(len(toks), 1))

        total_text_tokens = sum(text_token_counts)

        seg_token_ranges: List[Tuple[int, int]] = []
        cursor = 0
        for i, tc in enumerate(text_token_counts):
            count = round(total_tokens * tc / total_text_tokens)
            count = max(count, 1)
            if i == len(text_token_counts) - 1:
                count = total_tokens - cursor  # give remainder to last
            seg_token_ranges.append((cursor, cursor + count))
            cursor += count

        # ── 4. Resample each speech segment's tokens ────────────────────
        #
        # Build a mapping: speech_segment → (original_tokens, resampled_tokens)
        speech_seg_data = {}  # id(seg) → dict
        for seg, (tok_start, tok_end) in zip(speech_segments, seg_token_ranges):
            orig_tokens = all_tokens[tok_start:tok_end]
            natural_dur = len(orig_tokens) * SECS_PER_TOKEN
            target_dur = seg.target_duration

            if target_dur is not None and target_dur > 0:
                target_token_count = max(1, round(target_dur / SECS_PER_TOKEN))
                resampled = _resample_tokens(orig_tokens, target_token_count)
                actual_dur = len(resampled) * SECS_PER_TOKEN
                comfort = _comfort_score(natural_dur, target_dur, comfort_steepness)
            else:
                resampled = orig_tokens
                actual_dur = natural_dur
                comfort = 0.5

            speech_seg_data[id(seg)] = dict(
                orig_tokens=orig_tokens,
                resampled=resampled,
                natural_dur=natural_dur,
                actual_dur=actual_dur,
                comfort=comfort,
            )

            logger.info(
                "  Seg [%s…]: %d→%d tokens (%.2fs→%.2fs) comfort=%.2f",
                seg.text[:30], len(orig_tokens), len(resampled),
                natural_dur, actual_dur, comfort,
            )

        # ── 5. Group contiguous speech segments around silence gaps ──────
        #
        # Each "group" is a run of consecutive speech segments with no
        # silence between them.  Each group gets ONE S3Gen pass.
        # Silence gaps are inserted at the waveform level (just zeros).

        groups: List[List[TimedSegment]] = []
        current_group: List[TimedSegment] = []

        for seg in segments:
            if seg.is_silence:
                if current_group:
                    groups.append(current_group)
                    current_group = []
            else:
                current_group.append(seg)
        if current_group:
            groups.append(current_group)

        # ── 6. S3Gen pass per group ──────────────────────────────────────

        group_wavs = {}  # id(group[0]) → np.ndarray
        for group in groups:
            # Concatenate resampled tokens for all segments in this group
            group_tokens = torch.cat(
                [speech_seg_data[id(seg)]["resampled"] for seg in group]
            )
            logger.info(
                "S3Gen: %d tokens for group starting at %.2fs",
                len(group_tokens), group[0].start_time,
            )
            wav_tensor = self.model.speech_tokens_to_wav(
                group_tokens, apply_watermark=False
            )
            group_wav_np = wav_tensor.squeeze(0).numpy().astype(np.float32)
            group_wavs[id(group[0])] = group_wav_np

        # ── 7. Assemble final audio with absolute positioning ────────────

        # Determine which segments border silence (for fades)
        borders_silence_after = set()
        borders_silence_before = set()
        for idx, seg in enumerate(segments):
            if seg.is_silence:
                if idx > 0 and not segments[idx - 1].is_silence:
                    borders_silence_after.add(id(segments[idx - 1]))
                if idx + 1 < len(segments) and not segments[idx + 1].is_silence:
                    borders_silence_before.add(id(segments[idx + 1]))

        # Calculate total output length
        last_seg = segments[-1]
        if last_seg.is_silence:
            total_out_samples = int(round(
                (last_seg.end_time or last_seg.start_time) * self.sr
            ))
        elif last_seg.end_time is not None:
            total_out_samples = int(round(last_seg.end_time * self.sr))
        else:
            d = speech_seg_data[id(last_seg)]
            total_out_samples = (
                int(round(last_seg.start_time * self.sr))
                + int(round(d["actual_dur"] * self.sr))
            )

        final_wav = np.zeros(total_out_samples, dtype=np.float32)
        segment_results: List[SegmentResult] = []

        # Place each group's audio, slicing per-segment within the group
        for group in groups:
            group_audio = group_wavs[id(group[0])]
            sample_cursor = 0  # position within this group's audio

            for seg in group:
                d = speech_seg_data[id(seg)]
                seg_samples = len(d["resampled"]) * SAMPLES_PER_TOKEN
                seg_audio = group_audio[sample_cursor:sample_cursor + seg_samples]
                sample_cursor += seg_samples

                # Apply silence-boundary fades
                if id(seg) in borders_silence_after:
                    seg_audio = _apply_fade_out(seg_audio, self.sr)
                if id(seg) in borders_silence_before:
                    seg_audio = _apply_fade_in(seg_audio, self.sr)

                # Place at absolute position
                write_start = int(round(seg.start_time * self.sr))
                write_end = min(write_start + len(seg_audio), total_out_samples)
                n = write_end - write_start
                if n > 0:
                    final_wav[write_start:write_end] = seg_audio[:n]

                segment_results.append(SegmentResult(
                    text=seg.text,
                    start_time=seg.start_time,
                    end_time=seg.start_time + d["actual_dur"],
                    target_duration=seg.target_duration,
                    natural_duration=round(d["natural_dur"], 4),
                    actual_duration=round(d["actual_dur"], 4),
                    tokens_original=len(d["orig_tokens"]),
                    tokens_resampled=len(d["resampled"]),
                    comfort=round(d["comfort"], 3),
                ))

        # Add silence segment results
        for seg in segments:
            if seg.is_silence:
                dur = seg.target_duration or 0.0
                segment_results.append(SegmentResult(
                    text="", start_time=seg.start_time,
                    end_time=seg.end_time or seg.start_time,
                    target_duration=seg.target_duration,
                    natural_duration=0.0, actual_duration=dur,
                    tokens_original=0, tokens_resampled=0,
                    comfort=0.5,
                ))

        # Sort results by start_time (speech + silence interleaved)
        segment_results.sort(key=lambda r: r.start_time)

        # Apply watermark once on the complete assembled audio
        final_wav = self.model.watermarker.apply_watermark(
            final_wav, sample_rate=self.sr
        )

        final_tensor = torch.from_numpy(final_wav).unsqueeze(0)
        return TimedResult(
            wav=final_tensor,
            sr=self.sr,
            segments=segment_results,
            total_duration=round(len(final_wav) / self.sr, 4),
        )

    @staticmethod
    def parse(timed_text: str) -> List[TimedSegment]:
        return parse_timed_text(timed_text)


# ── Pretty-print ─────────────────────────────────────────────────────────────

def print_comfort_report(results: List[SegmentResult]) -> None:
    print()
    print("  Timed TTS — Segment Report")
    print("  " + "=" * 90)
    print(
        f"  {'#':>2}  {'Comfort':>7}  {'Natural':>8}  {'Actual':>8}  "
        f"{'Target':>8}  {'Tokens':>12}  Text"
    )
    print("  " + "-" * 90)

    for i, r in enumerate(results):
        label = "[SILENCE]" if not r.text else (
            r.text[:38] + ("…" if len(r.text) > 38 else "")
        )
        tgt = f"{r.target_duration:.2f}s" if r.target_duration is not None else "free"
        tok = f"{r.tokens_original}→{r.tokens_resampled}" if r.tokens_original else "-"

        tag = ""
        if r.comfort < 0.15:
            tag = " !! TOO SLOW"
        elif r.comfort < 0.30:
            tag = " ! slow"
        elif r.comfort > 0.85:
            tag = " !! TOO FAST"
        elif r.comfort > 0.70:
            tag = " ! fast"

        print(
            f"  {i:2d}  {r.comfort:7.2f}  {r.natural_duration:7.2f}s  "
            f"{r.actual_duration:7.2f}s  {tgt:>8}  {tok:>12}  {label}{tag}"
        )

    print("  " + "=" * 90)
    print()