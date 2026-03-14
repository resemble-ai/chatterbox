# src/chatterbox/timed_tts.py
#
# Timed narration for Chatterbox TTS.
#
# Architecture:
#   - One T3 pass **per silence-separated group** so the model naturally
#     plans sentence-ending prosody at every pause boundary.
#   - Token resampling per segment to hit target durations.
#   - A **single S3Gen pass** on all concatenated resampled tokens so the
#     vocoder sees one continuous stream → coherent voice, no startup-mute
#     artifacts between groups.
#   - Silence gaps are inserted at the waveform level via absolute positioning.
#
# Why per-group T3 instead of one giant T3 pass?
#   When T3 receives "And here we go! Back to action!" as one string, it
#   generates continuous speech with no prosodic break between the two
#   sentences.  A proportional token-split then lands mid-phoneme,
#   creating "And here we go! Ba..." | silence | "...ck to action!".
#   By giving T3 each group separately, it naturally winds down at "go!"
#   and starts fresh at "Back" — the split is clean because the token
#   streams are physically separate.
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
import os
import sys
import contextlib

import math
import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from .tts import ChatterboxTTS, punc_norm
from .models.s3gen.const import S3GEN_SR, TOKEN_TO_WAV_RATIO, TOKEN_MEL_RATIO

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

SECS_PER_TOKEN = TOKEN_TO_WAV_RATIO / S3GEN_SR    # 0.04 s
SAMPLES_PER_TOKEN = TOKEN_TO_WAV_RATIO             # 960
MEL_FRAMES_PER_TOKEN = TOKEN_MEL_RATIO             # 2
SAMPLES_PER_MEL_FRAME = TOKEN_TO_WAV_RATIO // TOKEN_MEL_RATIO  # 480
MEL_FRAMES_PER_SEC = S3GEN_SR / SAMPLES_PER_MEL_FRAME          # 50.0

# Comfort estimation heuristic: average speech tokens per text token.
_AVG_SPEECH_TOKENS_PER_TEXT_TOKEN = 2.1

# Fade duration (ms) at silence-gap boundaries in the final waveform.
_SILENCE_FADE_MS = 60


# ── Timing-tag regex ─────────────────────────────────────────────────────────

_TAG_RE = re.compile(r"<(\d+(?:\.\d+)?)>")


@contextlib.contextmanager
def _suppress_stdout():
    """Redirect stdout/stderr to devnull and silence chatterbox loggers below WARNING."""
    with open(os.devnull, 'w', encoding='utf-8') as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        # Mute chatterbox INFO/DEBUG that goes through the logging system
        cb_logger = logging.getLogger('chatterbox')
        old_level = cb_logger.level
        cb_logger.setLevel(logging.WARNING)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            cb_logger.setLevel(old_level)


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


# ── Mel-level resampling ─────────────────────────────────────────────────────

def _resample_mel(mel: torch.Tensor, target_mel_frames: int) -> torch.Tensor:
    """
    Smoothly resize a mel spectrogram slice along the time axis.

    mel : shape ``(channels, mel_frames)``
    Returns ``(channels, target_mel_frames)``.

    Linear interpolation on continuous mel values produces smooth
    time-stretching — no discrete-token artifacts.
    """
    n = mel.size(1)
    if n == 0 or target_mel_frames <= 0:
        return mel[:, :0]
    if target_mel_frames == n:
        return mel
    out = torch.nn.functional.interpolate(
        mel.unsqueeze(0),                     # (1, 80, n)
        size=target_mel_frames,
        mode='linear',
        align_corners=False,
    )
    return out.squeeze(0)                     # (80, target_mel_frames)


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


# ── Grouping helper ──────────────────────────────────────────────────────────

def _group_by_silence(segments: List[TimedSegment]) -> List[List[TimedSegment]]:
    """
    Split the flat segment list into groups of contiguous speech segments.
    Silence segments act as group separators and are not included in any group.
    """
    groups: List[List[TimedSegment]] = []
    current: List[TimedSegment] = []
    for seg in segments:
        if seg.is_silence:
            if current:
                groups.append(current)
                current = []
        else:
            current.append(seg)
    if current:
        groups.append(current)
    return groups


# ── Main class ───────────────────────────────────────────────────────────────

class TimedChatterboxTTS:
    """
    Timing-aware wrapper around :class:`ChatterboxTTS`.

    Pipeline:
      1. Parse ``<timestamp>`` tags → segments with target durations.
      2. Group contiguous speech segments (silence = group boundary).
      3. **One T3 pass per group** → natural sentence-ending prosody at
         every silence boundary.  Within a group, prosody flows naturally.
      4. Proportional token split within each group → per-segment tokens.
      5. **Resample** each segment's tokens to hit its target duration.
      6. Concatenate ALL resampled tokens from all groups.
      7. **Single S3Gen pass** → one coherent waveform (no startup-mute
         artifacts between groups).
      8. Assemble final audio with absolute positioning + silence gaps.
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
        quiet: bool = False,
    ) -> TimedResult:
        """
        Generate time-aligned narration.

        Per-group T3 passes → token resampling → single S3Gen pass.
        Silence gaps are handled at the waveform level.
        """

        # ── 1. Parse ─────────────────────────────────────────────────────
        segments = parse_timed_text(timed_text)
        if not segments:
            raise ValueError("No segments found.")

        speech_segments = [s for s in segments if not s.is_silence]
        if not speech_segments:
            raise ValueError("All segments empty.")

        # ── 2. Group contiguous speech segments around silence gaps ──────
        #
        # Each group is a run of speech segments with no silence between
        # them.  Each group gets its own T3 call so the model naturally
        # plans sentence-ending prosody at silence boundaries.
        groups = _group_by_silence(segments)

        if audio_prompt_path:
            self.model.prepare_conditionals(
                audio_prompt_path, exaggeration=exaggeration
            )

        t3_kwargs = dict(
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        )

        # ── 3. T3 pass per group + proportional split within group ──────
        #
        # For each group we:
        #   a) Join the group's segment texts into one string.
        #   b) Run T3 → get speech tokens with natural ending prosody.
        #   c) Proportionally split the group's tokens across its segments.
        speech_seg_data = {}  # id(seg) → dict

        for group in groups:
            group_text = " ".join(seg.text for seg in group)
            logger.debug("T3 generating group: %.120s…", group_text)

            if quiet:
                with _suppress_stdout():
                    group_tokens = self.model.generate_speech_tokens(
                        group_text, **t3_kwargs
                    )
            else:
                group_tokens = self.model.generate_speech_tokens(
                    group_text, **t3_kwargs
                )

            # group_tokens: 1-D LongTensor on CPU
            group_total = len(group_tokens)
            logger.debug(
                "  T3 produced %d tokens (%.2fs natural).",
                group_total, group_total * SECS_PER_TOKEN,
            )

            # ── Proportional split within this group ─────────────────
            text_token_counts = []
            for seg in group:
                normed = punc_norm(seg.text)
                toks = self.model.tokenizer.text_to_tokens(normed).squeeze(0)
                text_token_counts.append(max(len(toks), 1))

            total_text_tokens = sum(text_token_counts)

            seg_token_ranges: List[Tuple[int, int]] = []
            cursor = 0
            n_segs = len(text_token_counts)
            for i, tc in enumerate(text_token_counts):
                if i == n_segs - 1:
                    count = group_total - cursor
                else:
                    count = round(group_total * tc / total_text_tokens)
                    count = max(count, 1)
                    # Don't consume so many tokens that later segments starve.
                    remaining_after = n_segs - 1 - i   # segments still to come
                    count = min(count, group_total - cursor - remaining_after)
                    count = max(count, 1)
                seg_token_ranges.append((cursor, cursor + count))
                cursor += count

            # ── 4. Compute comfort + target mel frames per segment ───
            for seg, (tok_start, tok_end) in zip(group, seg_token_ranges):
                orig_tokens = group_tokens[tok_start:tok_end]
                natural_dur = len(orig_tokens) * SECS_PER_TOKEN
                target_dur = seg.target_duration

                if target_dur is not None and target_dur > 0:
                    comfort = _comfort_score(natural_dur, target_dur, comfort_steepness)
                    stretched_mel_frames = max(1, round(target_dur * MEL_FRAMES_PER_SEC))
                    actual_dur = stretched_mel_frames / MEL_FRAMES_PER_SEC
                else:
                    comfort = 0.5
                    stretched_mel_frames = len(orig_tokens) * MEL_FRAMES_PER_TOKEN
                    actual_dur = natural_dur

                speech_seg_data[id(seg)] = dict(
                    orig_tokens=orig_tokens,
                    natural_dur=natural_dur,
                    actual_dur=actual_dur,
                    comfort=comfort,
                    stretched_mel_frames=stretched_mel_frames,
                )

                logger.debug(
                    "  Seg [%s…]: %d tokens (%.2fs→%.2fs) comfort=%.2f",
                    seg.text[:30], len(orig_tokens),
                    natural_dur, actual_dur, comfort,
                )

        # ── 5. Flow on natural tokens → mel → interpolate → HiFi-GAN ────
        #
        # Run the flow model on the NATURAL (un-resampled) token stream
        # so it produces a clean mel spectrogram.  Then do time-stretching
        # at the mel level (linear interpolation on continuous floats) —
        # no discrete-token artifacts.  Finally one HiFi-GAN pass on the
        # concatenated stretched mel → one coherent waveform.

        all_natural = torch.cat(
            [speech_seg_data[id(seg)]["orig_tokens"] for seg in speech_segments]
        )
        logger.debug("Flow: %d natural tokens → mel", len(all_natural))

        if quiet:
            with _suppress_stdout():
                full_mel = self.model.speech_tokens_to_mel(all_natural)
        else:
            full_mel = self.model.speech_tokens_to_mel(all_natural)

        full_mel = full_mel.squeeze(0)  # (80, total_mel_frames)

        # Slice mel per segment, interpolate to target, concatenate
        mel_cursor = 0
        stretched_mel_slices = []

        for seg in speech_segments:
            d = speech_seg_data[id(seg)]
            natural_mel_frames = len(d["orig_tokens"]) * MEL_FRAMES_PER_TOKEN
            mel_slice = full_mel[:, mel_cursor:mel_cursor + natural_mel_frames]
            mel_cursor += natural_mel_frames
            stretched = _resample_mel(mel_slice, d["stretched_mel_frames"])
            stretched_mel_slices.append(stretched)

        all_stretched_mel = torch.cat(stretched_mel_slices, dim=1)  # (80, T)
        logger.debug("HiFi-GAN: %d stretched mel frames", all_stretched_mel.size(1))

        if quiet:
            with _suppress_stdout():
                full_wav_tensor = self.model.mel_to_wav(
                    all_stretched_mel.unsqueeze(0)
                )
        else:
            full_wav_tensor = self.model.mel_to_wav(
                all_stretched_mel.unsqueeze(0)
            )
        full_wav_np = full_wav_tensor.squeeze(0).numpy()

        # ── 6. Assemble final audio with absolute positioning ────────────

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

        # Slice per-segment from the single S3Gen output and place at
        # absolute positions.  sample_cursor tracks our position within
        # the continuous full_wav_np.
        sample_cursor = 0

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
                continue

            d = speech_seg_data[id(seg)]
            seg_samples = d["stretched_mel_frames"] * SAMPLES_PER_MEL_FRAME
            seg_audio = full_wav_np[sample_cursor:sample_cursor + seg_samples]
            sample_cursor += seg_samples

            # Apply silence-boundary fades
            if id(seg) in borders_silence_after:
                seg_audio = _apply_fade_out(seg_audio, self.sr)
            if id(seg) in borders_silence_before:
                seg_audio = _apply_fade_in(seg_audio, self.sr)

            # If audio overflows the target window, truncate with fade-out
            if seg.end_time is not None:
                max_samples = int(round(seg.target_duration * self.sr))
                if len(seg_audio) > max_samples and max_samples > 0:
                    fade_len = min(
                        int(self.sr * _SILENCE_FADE_MS / 1000),
                        max_samples
                    )
                    seg_audio = seg_audio.copy()
                    if fade_len > 1:
                        t = np.linspace(0, np.pi / 2, fade_len, dtype=np.float32)
                        seg_audio[max_samples - fade_len:max_samples] *= np.cos(t) ** 2
                    seg_audio = seg_audio[:max_samples]

            # Place at absolute position
            write_start = int(round(seg.start_time * self.sr))
            write_end = min(write_start + len(seg_audio), total_out_samples)
            n = write_end - write_start
            if n > 0:
                final_wav[write_start:write_end] = seg_audio[:n]

            actual_dur = len(seg_audio) / self.sr

            segment_results.append(SegmentResult(
                text=seg.text,
                start_time=seg.start_time,
                end_time=seg.start_time + actual_dur,
                target_duration=seg.target_duration,
                natural_duration=round(d["natural_dur"], 4),
                actual_duration=round(actual_dur, 4),
                tokens_original=len(d["orig_tokens"]),
                tokens_resampled=d["stretched_mel_frames"] // MEL_FRAMES_PER_TOKEN,
                comfort=round(d["comfort"], 3),
            ))

        # Sort results by start_time (speech + silence interleaved)
        segment_results.sort(key=lambda r: r.start_time)

        # Apply watermark once on the complete assembled audio
        if self.model.apply_watermark:
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

    # ── Batched generation ─────────────────────────────────────────────
 
    def generate_batch(
        self,
        timed_texts: List[str],
        audio_prompt_path: Union[str, List[str]] = None,
        exaggeration: Union[float, List[float]] = 0.5,
        cfg_weight: Union[float, List[float]] = 0.5,
        temperature: Union[float, List[float]] = 0.8,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        *,
        comfort_steepness: float = 3.5,
        quiet: bool = False,
    ) -> List[List[TimedResult]]:
        """
        Generate time-aligned narration for multiple scripts in one batched pass.
 
        Parameters
        ----------
        timed_texts : List[str]
            List of scripts with ``<float>`` timing tags.
        exaggeration : float or List[float]
            Emotion intensity.  Single value applied to all scripts, or one
            per script for per-item control.
        cfg_weight : float or List[float]
            Classifier-free guidance weight.  Single or per-script.
        temperature : float or List[float]
            Sampling temperature.  Single or per-script.
        num_return_sequences : int
            Number of stochastic variants per script (default 1).
 
        Returns
        -------
        List[List[TimedResult]]
            Outer list = scripts, inner list = variants.
            ``result[script_idx][variant_idx].wav`` is a ``(1, N)`` tensor.
 
        Pipeline
        --------
        1. Parse + group all scripts.
        2. Flatten all group texts across all scripts.
        3. **Single batched T3 call** (with ``num_return_sequences``).
        4. Distribute tokens back per (script, variant, group).
        5. Proportional split + resample per segment.
        6. **Single batched S3Gen call** on all (script x variant) token streams.
        7. Assemble waveforms with absolute positioning.
        """
        N = num_return_sequences
        n_scripts = len(timed_texts)
 
        # ── 1. Parse + group all scripts ──────────────────────────────────
        all_segments: List[List[TimedSegment]] = []
        all_groups: List[List[List[TimedSegment]]] = []
 
        for raw in timed_texts:
            segments = parse_timed_text(raw)
            if not segments:
                raise ValueError(f"No segments found in: {raw[:80]}")
            speech = [s for s in segments if not s.is_silence]
            if not speech:
                raise ValueError(f"All segments empty in: {raw[:80]}")
            all_segments.append(segments)
            all_groups.append(_group_by_silence(segments))
 
        # ── Normalise per-script params to lists ─────────────────────────
        if isinstance(exaggeration, (int, float)):
            exaggeration = [float(exaggeration)] * n_scripts
        if isinstance(cfg_weight, (int, float)):
            cfg_weight = [float(cfg_weight)] * n_scripts
        if isinstance(temperature, (int, float)):
            temperature = [float(temperature)] * n_scripts
 
        # ── 2. Flatten group texts with tracking ─────────────────────────
        flat_group_texts: List[str] = []
        flat_group_exagg: List[float] = []
        flat_group_cfg: List[float] = []
        flat_group_temp: List[float] = []
        group_origin: List[Tuple[int, int]] = []
 
        for s_idx, groups in enumerate(all_groups):
            for g_idx, group in enumerate(groups):
                group_text = " ".join(seg.text for seg in group)
                flat_group_texts.append(group_text)
                flat_group_exagg.append(exaggeration[s_idx])
                flat_group_cfg.append(cfg_weight[s_idx])
                flat_group_temp.append(temperature[s_idx])
                group_origin.append((s_idx, g_idx))
 
        n_flat_groups = len(flat_group_texts)
 
        # ── 3. Prepare conditioning (once) ────────────────────────────────
        if audio_prompt_path:
            self.model.prepare_conditionals(
                audio_prompt_path, exaggeration=exaggeration[0]
            )
 
        # ── 4. Single batched T3 call ─────────────────────────────────────
        #
        # With num_return_sequences=N, T3 returns n_flat_groups * N items.
        # Ordering (due to repeat_interleave): consecutive N results per group:
        #   [g0_v0, g0_v1, ..., g0_vN-1, g1_v0, g1_v1, ...]
        t3_kwargs = dict(
            exaggeration=flat_group_exagg,
            cfg_weight=flat_group_cfg,
            temperature=flat_group_temp,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
            num_return_sequences=N,
        )
 
        logger.debug(
            "Batched T3: %d group texts x %d variants = %d items",
            n_flat_groups, N, n_flat_groups * N,
        )
 
        if quiet:
            with _suppress_stdout():
                all_group_tokens = self.model.generate_speech_tokens_batch(
                    flat_group_texts, **t3_kwargs
                )
        else:
            all_group_tokens = self.model.generate_speech_tokens_batch(
                flat_group_texts, **t3_kwargs
            )
 
        # ── 5. Distribute tokens back + resample ─────────────────────────
 
        def _get_group_tokens(flat_g_idx: int, v_idx: int) -> torch.Tensor:
            return all_group_tokens[flat_g_idx * N + v_idx]
 
        # Map script_idx -> list of flat group indices
        script_flat_indices: List[List[int]] = [[] for _ in range(n_scripts)]
        for flat_idx, (s_idx, _g_idx) in enumerate(group_origin):
            script_flat_indices[s_idx].append(flat_idx)
 
        # Per-(script, variant) resampled token streams + segment data
        sv_streams: List[List[Optional[torch.Tensor]]] = [
            [None] * N for _ in range(n_scripts)
        ]
        sv_seg_data: List[List[Optional[dict]]] = [
            [None] * N for _ in range(n_scripts)
        ]
 
        for s_idx in range(n_scripts):
            segments = all_segments[s_idx]
            groups = all_groups[s_idx]
            speech_segs = [s for s in segments if not s.is_silence]
            flat_indices = script_flat_indices[s_idx]
 
            for v_idx in range(N):
                seg_data = {}
 
                for local_g_idx, group in enumerate(groups):
                    flat_g_idx = flat_indices[local_g_idx]
                    group_tokens = _get_group_tokens(flat_g_idx, v_idx)
                    group_total = len(group_tokens)
 
                    # Proportional split within group
                    text_token_counts = []
                    for seg in group:
                        normed = punc_norm(seg.text)
                        toks = self.model.tokenizer.text_to_tokens(
                            normed
                        ).squeeze(0)
                        text_token_counts.append(max(len(toks), 1))
 
                    total_text_tokens = sum(text_token_counts)
                    n_segs = len(text_token_counts)
                    cursor = 0
 
                    for i, (seg, tc) in enumerate(
                        zip(group, text_token_counts)
                    ):
                        if i == n_segs - 1:
                            count = group_total - cursor
                        else:
                            count = round(
                                group_total * tc / total_text_tokens
                            )
                            count = max(count, 1)
                            # Don't consume so many tokens that later
                            # segments starve.
                            remaining_after = n_segs - 1 - i
                            count = min(
                                count,
                                group_total - cursor - remaining_after,
                            )
                            count = max(count, 1)
 
                        orig_tokens = group_tokens[cursor : cursor + count]
                        cursor += count
 
                        natural_dur = len(orig_tokens) * SECS_PER_TOKEN
                        target_dur = seg.target_duration
 
                        if target_dur is not None and target_dur > 0:
                            comfort = _comfort_score(
                                natural_dur, target_dur, comfort_steepness
                            )
                            stretched_mel_frames = max(
                                1, round(target_dur * MEL_FRAMES_PER_SEC)
                            )
                            actual_dur = stretched_mel_frames / MEL_FRAMES_PER_SEC
                        else:
                            comfort = 0.5
                            stretched_mel_frames = (
                                len(orig_tokens) * MEL_FRAMES_PER_TOKEN
                            )
                            actual_dur = natural_dur

                        seg_data[id(seg)] = dict(
                            orig_tokens=orig_tokens,
                            natural_dur=natural_dur,
                            actual_dur=actual_dur,
                            comfort=comfort,
                            stretched_mel_frames=stretched_mel_frames,
                        )
 
                sv_streams[s_idx][v_idx] = torch.cat(
                    [seg_data[id(seg)]["orig_tokens"] for seg in speech_segs]
                )
                sv_seg_data[s_idx][v_idx] = seg_data
 
        # ── 6. Batched flow → mel interpolation → batched HiFi-GAN ──────
        flat_streams: List[torch.Tensor] = []
        flat_sv_index: List[Tuple[int, int]] = []
        for s_idx in range(n_scripts):
            for v_idx in range(N):
                flat_streams.append(sv_streams[s_idx][v_idx])
                flat_sv_index.append((s_idx, v_idx))

        logger.debug(
            "Batched flow: %d streams (max %d tokens)",
            len(flat_streams),
            max(len(s) for s in flat_streams),
        )

        # 6a. Batched flow: natural tokens → mel per (script, variant)
        if quiet:
            with _suppress_stdout():
                mel_list = self.model.speech_tokens_to_mel_batch(flat_streams)
        else:
            mel_list = self.model.speech_tokens_to_mel_batch(flat_streams)

        # 6b. Per-(script, variant) mel slicing + interpolation
        stretched_mel_list: List[torch.Tensor] = []
        for flat_idx, (s_idx, v_idx) in enumerate(flat_sv_index):
            segments = all_segments[s_idx]
            speech_segs = [s for s in segments if not s.is_silence]
            seg_data = sv_seg_data[s_idx][v_idx]
            full_mel = mel_list[flat_idx]  # (80, mel_frames)

            mel_cursor = 0
            slices = []
            for seg in speech_segs:
                d = seg_data[id(seg)]
                nat_mel = len(d["orig_tokens"]) * MEL_FRAMES_PER_TOKEN
                mel_slice = full_mel[:, mel_cursor:mel_cursor + nat_mel]
                mel_cursor += nat_mel
                slices.append(_resample_mel(mel_slice, d["stretched_mel_frames"]))

            stretched_mel_list.append(torch.cat(slices, dim=1))

        # 6c. Batched HiFi-GAN: stretched mels → wavs
        logger.debug(
            "Batched HiFi-GAN: %d streams (max %d mel frames)",
            len(stretched_mel_list),
            max(m.size(1) for m in stretched_mel_list),
        )

        if quiet:
            with _suppress_stdout():
                wav_list = self.model.mel_to_wav_batch(
                    stretched_mel_list, apply_trim_fade=True
                )
        else:
            wav_list = self.model.mel_to_wav_batch(
                stretched_mel_list, apply_trim_fade=True
            )

        sv_full_wav: List[List[Optional[np.ndarray]]] = [
            [None] * N for _ in range(n_scripts)
        ]
        for flat_idx, (s_idx, v_idx) in enumerate(flat_sv_index):
            sv_full_wav[s_idx][v_idx] = wav_list[flat_idx].squeeze(0).numpy()
 
        # ── 7. Assemble waveforms per (script, variant) ───────────────────
        all_results: List[List[TimedResult]] = []
 
        for s_idx in range(n_scripts):
            segments = all_segments[s_idx]
            speech_segs = [s for s in segments if not s.is_silence]
            variant_results: List[TimedResult] = []
 
            # Determine which segments border silence (for fades)
            borders_silence_after = set()
            borders_silence_before = set()
            for idx, seg in enumerate(segments):
                if seg.is_silence:
                    if idx > 0 and not segments[idx - 1].is_silence:
                        borders_silence_after.add(id(segments[idx - 1]))
                    if idx + 1 < len(segments) and not segments[idx + 1].is_silence:
                        borders_silence_before.add(id(segments[idx + 1]))
 
            for v_idx in range(N):
                seg_data = sv_seg_data[s_idx][v_idx]
                full_wav_np = sv_full_wav[s_idx][v_idx]
 
                # Calculate total output length
                last_seg = segments[-1]
                if last_seg.is_silence:
                    total_out_samples = int(round(
                        (last_seg.end_time or last_seg.start_time) * self.sr
                    ))
                elif last_seg.end_time is not None:
                    total_out_samples = int(round(last_seg.end_time * self.sr))
                else:
                    d = seg_data[id(last_seg)]
                    total_out_samples = (
                        int(round(last_seg.start_time * self.sr))
                        + int(round(d["actual_dur"] * self.sr))
                    )
 
                final_wav = np.zeros(total_out_samples, dtype=np.float32)
                segment_results: List[SegmentResult] = []
 
                # Slice per-segment from the single S3Gen output and place
                # at absolute positions.  sample_cursor tracks our position
                # within the continuous full_wav_np.
                sample_cursor = 0
 
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
                        continue
 
                    d = seg_data[id(seg)]
                    seg_samples = d["stretched_mel_frames"] * SAMPLES_PER_MEL_FRAME
                    seg_audio = full_wav_np[sample_cursor:sample_cursor + seg_samples]
                    sample_cursor += seg_samples
 
                    # Apply silence-boundary fades
                    if id(seg) in borders_silence_after:
                        seg_audio = _apply_fade_out(seg_audio, self.sr)
                    if id(seg) in borders_silence_before:
                        seg_audio = _apply_fade_in(seg_audio, self.sr)
 
                    # If audio overflows the target window, truncate with
                    # fade-out
                    if seg.end_time is not None:
                        max_samples = int(round(seg.target_duration * self.sr))
                        if len(seg_audio) > max_samples and max_samples > 0:
                            fade_len = min(
                                int(self.sr * _SILENCE_FADE_MS / 1000),
                                max_samples,
                            )
                            seg_audio = seg_audio.copy()
                            if fade_len > 1:
                                t = np.linspace(
                                    0, np.pi / 2, fade_len, dtype=np.float32
                                )
                                seg_audio[max_samples - fade_len:max_samples] *= (
                                    np.cos(t) ** 2
                                )
                            seg_audio = seg_audio[:max_samples]
 
                    # Place at absolute position
                    write_start = int(round(seg.start_time * self.sr))
                    write_end = min(
                        write_start + len(seg_audio), total_out_samples
                    )
                    n = write_end - write_start
                    if n > 0:
                        final_wav[write_start:write_end] = seg_audio[:n]
 
                    actual_dur = len(seg_audio) / self.sr
                    segment_results.append(SegmentResult(
                        text=seg.text,
                        start_time=seg.start_time,
                        end_time=seg.start_time + actual_dur,
                        target_duration=seg.target_duration,
                        natural_duration=round(d["natural_dur"], 4),
                        actual_duration=round(actual_dur, 4),
                        tokens_original=len(d["orig_tokens"]),
                        tokens_resampled=d["stretched_mel_frames"] // MEL_FRAMES_PER_TOKEN,
                        comfort=round(d["comfort"], 3),
                    ))
 
                # Sort results by start_time (speech + silence interleaved)
                segment_results.sort(key=lambda r: r.start_time)
 
                # Apply watermark once on the complete assembled audio
                if self.model.apply_watermark:
                    final_wav = self.model.watermarker.apply_watermark(
                        final_wav, sample_rate=self.sr
                    )
 
                variant_results.append(TimedResult(
                    wav=torch.from_numpy(final_wav).unsqueeze(0),
                    sr=self.sr,
                    segments=segment_results,
                    total_duration=round(len(final_wav) / self.sr, 4),
                ))
 
            all_results.append(variant_results)
 
        return all_results


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