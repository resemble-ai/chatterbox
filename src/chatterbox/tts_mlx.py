# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
MLX-optimized TTS pipeline for Chatterbox.
Provides significant performance improvements on Apple Silicon (M1/M2/M3/M4).

This implementation uses a hybrid approach:
- MLX for T3 (autoregressive text-to-speech token generation) - main performance win
- PyTorch for S3Gen (vocoder) - uses MPS acceleration on Apple Silicon
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import logging
import re
import os

import numpy as np

# Import shared utilities for consistent behavior across all TTS implementations
from .generation_utils import (
    split_text_intelligently,
    crossfade_chunks,
    print_generation_plan,
    print_chunk_generating,
    print_chunk_completed,
    print_crossfading,
    print_generation_complete,
    SPACY_AVAILABLE,
    ADAPTIVE_THRESHOLD_WORDS,
    TARGET_WORDS_PER_CHUNK,
    split_into_sentences,
)

# Import memory utilities for debugging
from .models.utils import get_memory_info, is_debug

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    _mlx_import_error = (
        "MLX is not installed. Install it with:\n"
        "  pip install chatterbox-tts[mlx]\n"
        "or manually:\n"
        "  pip install mlx mlx-lm"
    )

if not MLX_AVAILABLE:
    raise ImportError(_mlx_import_error)

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.t3_mlx.t3_mlx import T3MLX
from .models.t3_mlx.modules.cond_enc_mlx import T3CondMLX
from .models.t3.modules.t3_config import T3Config
from .models.t3.modules.cond_enc import T3Cond

# Use PyTorch S3Gen for now (hybrid approach - T3 in MLX, S3Gen in PyTorch/MPS)
from .models.s3gen import S3Gen, S3GEN_SR
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.voice_encoder import VoiceEncoder
from .models.tokenizers import EnTokenizer

import perth
import librosa

logger = logging.getLogger(__name__)

REPO_ID = "ResembleAI/chatterbox"


def _log_memory_mlx(label: str):
    """
    Log detailed memory info for MLX debugging.
    Only logs when DEBUG_LOGGING or DEBUG_MEMORY env var is set.
    Uses get_memory_info() from chatterbox.models.utils.
    """
    if not is_debug() and os.environ.get("DEBUG_MEMORY", "0") != "1":
        return

    info = get_memory_info()
    parts = [f"[MLX MEM] {label}:"]
    parts.append(f"Sys={info['sys_used_gb']:.2f}GB ({info['sys_percent']:.0f}%)")

    if "wired_gb" in info:
        parts.append(f"Wired={info['wired_gb']:.2f}GB")
    if "active_gb" in info:
        parts.append(f"Active={info['active_gb']:.2f}GB")
    if "mps_allocated_mb" in info:
        parts.append(f"MPS={info['mps_allocated_mb']:.0f}MB")

    # Force MLX to sync
    mx.eval(mx.array([0]))

    logger.debug(" | ".join(parts))
    print(" | ".join(parts))


def punc_norm(text: str) -> str:
    """Quick cleanup for punctuation."""
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalize first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple spaces
    text = " ".join(text.split())

    # Replace uncommon punctuation
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        (""", "\""), (""", '"'),
        ("'", "'"),
        ("'", "'"),
    ]
    for old, new in punc_to_replace:
        text = text.replace(old, new)

    # Add full stop if no ending punctuation
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class ConditionalsMLX:
    """
    Conditionals for T3 MLX and S3Gen MLX.
    """

    t3: T3CondMLX
    gen: dict  # Reference dict for S3Gen

    def to_device(self):
        """MLX uses unified memory, no-op for compatibility."""
        return self


class ChatterboxTTSMLX:
    """
    MLX-optimized Chatterbox TTS pipeline.

    Uses a hybrid approach for best performance on Apple Silicon:
    - MLX for T3 (autoregressive token generation) - main speedup
    - PyTorch/MPS for S3Gen (vocoder) - fast on Apple Silicon

    This provides 2-3x speedup compared to pure PyTorch MPS backend.
    """

    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3MLX,
        s3gen: S3Gen,  # PyTorch S3Gen for now
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str = "mps",
        conds=None,  # Conditionals from tts.py
    ):
        """
        Initialize hybrid MLX/PyTorch TTS pipeline.

        Args:
            t3: T3MLX model (MLX)
            s3gen: S3Gen vocoder (PyTorch)
            ve: Voice encoder (PyTorch)
            tokenizer: Text tokenizer
            device: PyTorch device for S3Gen ("mps" or "cpu")
            conds: Optional pre-computed conditioning
        """
        self.sr = S3GEN_SR  # Sample rate: 24kHz
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()

        # Cached MLX conditioning (optimization #4 - avoid re-converting for each sentence)
        self._cached_t3_cond_mx: Optional[T3CondMLX] = None
        self._cached_cond_hash: Optional[int] = None

    def _get_cached_t3_cond_mx(self) -> T3CondMLX:
        """
        Get cached MLX conditioning, converting from PyTorch only if changed.

        This avoids redundant PyTorch->MLX conversion when generating multiple
        sentences with the same conditioning (e.g., in generate_long).
        """
        if self.conds is None:
            raise ValueError(
                "Conditioning not prepared. Call prepare_conditionals first."
            )

        # Compute a simple hash of conditioning to detect changes
        cond_hash = id(self.conds.t3) + hash(
            float(self.conds.t3.emotion_adv[0, 0, 0].item())
        )

        if self._cached_t3_cond_mx is None or self._cached_cond_hash != cond_hash:
            # Convert to MLX and cache
            self._cached_t3_cond_mx = T3CondMLX(
                speaker_emb=(
                    mx.array(self.conds.t3.speaker_emb.cpu().numpy())
                    if self.conds.t3.speaker_emb is not None
                    else None
                ),
                cond_prompt_speech_tokens=(
                    mx.array(self.conds.t3.cond_prompt_speech_tokens.cpu().numpy())
                    if self.conds.t3.cond_prompt_speech_tokens is not None
                    else None
                ),
                emotion_adv=float(self.conds.t3.emotion_adv[0, 0, 0].item()),
            )
            self._cached_cond_hash = cond_hash
            logger.debug("Cached MLX conditioning updated")

        return self._cached_t3_cond_mx

    @classmethod
    def from_pretrained(
        cls,
        cache_dir: Optional[str] = None,
        t3_config: Optional[T3Config] = None,
        use_default_speaker: bool = True,
        device: str = "mps",
    ) -> "ChatterboxTTSMLX":
        """
        Load pre-trained Chatterbox TTS model with MLX optimization.

        Args:
            cache_dir: Optional cache directory for model files
            t3_config: Optional T3 configuration (default: English-only)
            use_default_speaker: Whether to load default speaker conditioning
            device: PyTorch device for S3Gen ("mps" or "cpu")

        Returns:
            ChatterboxTTSMLX instance
        """
        logger.info("Loading Chatterbox TTS with MLX optimization...")

        # Check MPS availability
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS not available, falling back to CPU for S3Gen")
            device = "cpu"

        # Download model files from HuggingFace
        for fpath in [
            "ve.safetensors",
            "t3_cfg.safetensors",
            "s3gen.safetensors",
            "tokenizer.json",
            "conds.pt",
        ]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        ckpt_dir = Path(local_path).parent

        return cls.from_local(ckpt_dir, t3_config, use_default_speaker, device)

    @classmethod
    def from_local(
        cls,
        ckpt_dir: Path,
        t3_config: Optional[T3Config] = None,
        use_default_speaker: bool = True,
        device: str = "mps",
    ) -> "ChatterboxTTSMLX":
        """
        Load model from local checkpoint directory.

        Args:
            ckpt_dir: Path to checkpoint directory
            t3_config: Optional T3 configuration
            use_default_speaker: Whether to load default speaker
            device: PyTorch device for S3Gen

        Returns:
            ChatterboxTTSMLX instance
        """
        ckpt_dir = Path(ckpt_dir)

        # Check MPS availability
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS not available, falling back to CPU")
            device = "cpu"

        # Load voice encoder (PyTorch)
        logger.info("Loading voice encoder...")
        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve.to(device).eval()

        # Initialize T3 MLX model
        logger.info("Initializing T3 MLX model...")
        if t3_config is None:
            t3_config = T3Config.english_only()

        t3 = T3MLX(hp=t3_config)

        # Load T3 weights and convert to MLX
        logger.info("Loading T3 weights into MLX...")
        t3_ckpt = ckpt_dir / "t3_cfg.safetensors"
        pt_state = load_file(t3_ckpt)
        if "model" in pt_state.keys():
            pt_state = pt_state["model"][0]

        # Convert PyTorch weights to MLX format
        mlx_state = {}
        for k, v in pt_state.items():
            mlx_state[k] = mx.array(v.cpu().numpy())

        # Load weights into T3MLX with strict=False to ignore unused embed_tokens
        t3.load_weights(list(mlx_state.items()), strict=False)
        t3.eval()  # Set to eval mode to disable dropout
        logger.info("✓ T3 weights loaded successfully")

        # Load S3Gen (PyTorch/MPS) - hybrid approach
        logger.info(f"Loading S3Gen vocoder (PyTorch on {device})...")
        s3gen = S3Gen()
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
        s3gen.to(device).eval()
        logger.info("✓ S3Gen loaded successfully")

        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = EnTokenizer(str(ckpt_dir / "tokenizer.json"))

        # Load default conditioning if requested
        conds = None
        if use_default_speaker:
            logger.info("Loading default speaker conditioning...")
            cond_path = ckpt_dir / "conds.pt"
            if cond_path.exists():
                from .tts import Conditionals

                conds = Conditionals.load(cond_path, map_location=device)
                logger.info("✓ Default speaker conditioning loaded")

        logger.info("✅ ChatterboxTTSMLX loaded successfully!")
        logger.info("   T3: MLX (Apple Silicon optimized)")
        logger.info(f"   S3Gen: PyTorch on {device}")

        return cls(
            t3=t3,
            s3gen=s3gen,
            ve=ve,
            tokenizer=tokenizer,
            device=device,
            conds=conds,
        )

    def prepare_conditionals(self, wav_fpath: str, exaggeration: float = 0.5):
        """
        Prepare conditioning from a reference audio file.

        Args:
            wav_fpath: Path to reference audio file
            exaggeration: Emotion exaggeration factor (0.0 to 1.0)
        """
        from .tts import Conditionals

        # Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[: self.DEC_COND_LEN]

        # Get S3Gen reference embedding (PyTorch)
        s3gen_ref_dict = self.s3gen.embed_ref(
            s3gen_ref_wav, S3GEN_SR, device=self.device
        )

        # Speech cond prompt tokens for T3
        t3_cond_prompt_tokens = None
        if hasattr(self.t3, "hp") and self.t3.hp.speech_cond_prompt_len:
            plen = self.t3.hp.speech_cond_prompt_len
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward(
                [ref_16k_wav[: self.ENC_COND_LEN]], max_len=plen
            )
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(
                self.device
            )

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(
            self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR)
        )
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)

        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

        # Clear conditioning cache when conditioning changes
        self._cached_t3_cond_mx = None
        self._cached_cond_hash = None
        _log_memory_mlx("hybrid_conditionals_prepared")

    def _generate_single_sentence(
        self,
        text: str,
        cfg_weight: float = 0.5,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.8,
        top_p: float = 1.0,
        min_p: float = 0.05,
        repetition_penalty: float = 1.2,
        apply_watermark: bool = True,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate speech for a single sentence/chunk (internal method).

        Assumes conditioning is already prepared. This is the core generation
        logic extracted for use by both generate() and generate_long().

        Args:
            text: Input text (single sentence/chunk)
            cfg_weight: Classifier-free guidance weight
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            min_p: Minimum probability threshold
            repetition_penalty: Repetition penalty factor
            apply_watermark: Whether to apply watermark (default True, False for intermediate chunks)
            show_progress: Whether to show token-level progress bar

        Returns:
            Generated audio waveform as torch tensor
        """
        # Normalize and tokenize text
        text = punc_norm(text)

        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        # Estimate reasonable max_new_tokens based on text length if not provided
        # Speech tokens are roughly 10-15x text characters, with 2x safety buffer
        # This prevents runaway generation when EOS is not triggered
        if max_new_tokens is None:
            estimated_tokens = len(text) * 15  # ~15 speech tokens per character
            max_new_tokens = min(
                max(estimated_tokens * 2, 200),  # At least 200, 2x buffer
                self.t3.hp.max_speech_tokens,  # But never exceed model max
            )
            logger.debug(
                f"Estimated max_new_tokens: {max_new_tokens} for {len(text)} chars"
            )

        # Add start/end tokens for CFG
        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        # Convert to MLX for T3 inference
        text_tokens_mx = mx.array(text_tokens.cpu().numpy())

        # Get cached T3 conditioning (avoids re-conversion for each sentence)
        t3_cond_mx = self._get_cached_t3_cond_mx()

        _log_memory_mlx("hybrid_before_t3_inference")

        # Generate speech tokens with T3 MLX
        with torch.inference_mode():
            speech_tokens_mx = self.t3.inference(
                t3_cond=t3_cond_mx,
                text_tokens=text_tokens_mx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                show_progress=show_progress,
            )

        # Clear T3's KV cache to free memory after generation
        if hasattr(self.t3, "patched_model") and self.t3.patched_model is not None:
            self.t3.patched_model.reset_state()

        _log_memory_mlx("hybrid_after_t3_inference")

        # Extract conditional batch (first one)
        speech_tokens = (
            speech_tokens_mx[0] if len(speech_tokens_mx.shape) > 1 else speech_tokens_mx
        )

        # Convert back to PyTorch for S3Gen
        speech_tokens_np = np.array(speech_tokens).astype(np.int64)
        speech_tokens_pt = torch.from_numpy(speech_tokens_np)

        # Drop invalid tokens (SOS/EOS) - needs PyTorch tensor
        speech_tokens_pt = drop_invalid_tokens(speech_tokens_pt)

        # Filter out tokens >= 6561 (special tokens)
        speech_tokens_pt = speech_tokens_pt[speech_tokens_pt < 6561]
        speech_tokens_pt = speech_tokens_pt.to(self.device)

        _log_memory_mlx("hybrid_before_s3gen_inference")

        # Generate waveform with S3Gen (PyTorch/MPS)
        with torch.inference_mode():
            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens_pt,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()

            if apply_watermark:
                wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)

        _log_memory_mlx("hybrid_after_s3gen_inference")

        return torch.from_numpy(wav).unsqueeze(0)

    def generate(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.8,
        top_p: float = 1.0,
        min_p: float = 0.05,
        repetition_penalty: float = 1.2,
        use_sentence_chunking: bool = True,
        overlap_duration: float = 0.05,
        language: str = "en",
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate speech from text using hybrid MLX/PyTorch pipeline.

        By default, splits text into sentences using spacy for optimal performance
        on MLX hybrid backend, which is fastest for shorter texts.

        Args:
            text: Input text to synthesize
            audio_prompt_path: Optional path to reference audio for voice cloning
            exaggeration: Emotion exaggeration factor (0.0 to 1.0)
            cfg_weight: Classifier-free guidance weight
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            min_p: Minimum probability threshold
            repetition_penalty: Repetition penalty factor
            use_sentence_chunking: Whether to split text into sentences (default True for MLX performance)
            overlap_duration: Crossfade duration between sentences in seconds
            language: Language code for spacy sentence tokenization (e.g., "en", "de", "fr")
            show_progress: Whether to show token-level progress bar (default True)

        Returns:
            Generated audio waveform as torch tensor
        """
        import time as _time

        # Prepare conditioning if audio prompt provided
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert (
                self.conds is not None
            ), "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Split text into sentences for optimal MLX performance
        if use_sentence_chunking and SPACY_AVAILABLE:
            sentences = split_into_sentences(text, lang=language)
        else:
            sentences = [text]

        total_words = len(text.split())
        num_chunks = len(sentences)

        # Generate audio for each sentence
        if len(sentences) == 1:
            # Single sentence - generate directly with watermark
            print_generation_plan(total_words, sentences, "single chunk")
            print_chunk_generating(0, 1, sentences[0])
            gen_start = _time.time()

            result = self._generate_single_sentence(
                sentences[0],
                cfg_weight=cfg_weight,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                apply_watermark=True,
                show_progress=show_progress,  # Use caller's preference
            )

            gen_time = _time.time() - gen_start
            audio_duration = (
                result.shape[-1] / self.sr
                if hasattr(result, "shape")
                else len(result) / self.sr
            )
            print_chunk_completed(0, 1, gen_time, audio_duration)
            print_generation_complete(gen_time, audio_duration)

            return result

        # Multiple sentences - print overview and generate each with status updates
        print_generation_plan(total_words, sentences, "per-sentence")

        # Generate each chunk with status updates
        audio_chunks = []
        total_start = _time.time()

        for i, sentence in enumerate(sentences):
            print_chunk_generating(i, num_chunks, sentence)
            chunk_start = _time.time()

            # Don't apply watermark to intermediate chunks
            chunk_audio = self._generate_single_sentence(
                sentence,
                cfg_weight=cfg_weight,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                apply_watermark=False,
            )

            chunk_time = _time.time() - chunk_start
            chunk_duration = (
                chunk_audio.shape[-1] / self.sr
                if hasattr(chunk_audio, "shape")
                else len(chunk_audio) / self.sr
            )
            print_chunk_completed(i, num_chunks, chunk_time, chunk_duration)

            audio_chunks.append(chunk_audio)

            # Memory cleanup between chunks
            # Note: mx.clear_cache() is already called in T3's generate() and reset_state()
            import gc

            gc.collect()

            # Clear PyTorch MPS cache periodically (every 3 chunks)
            if i > 0 and i % 3 == 0 and torch.backends.mps.is_available():
                torch.mps.empty_cache()

        total_time = _time.time() - total_start

        # Crossfade chunks together
        print_crossfading(num_chunks)
        result = self._crossfade_chunks(audio_chunks, overlap_duration)

        # Apply watermark to final concatenated audio
        result_np = result.numpy() if isinstance(result, torch.Tensor) else result
        watermarked_result = self.watermarker.apply_watermark(
            result_np, sample_rate=self.sr
        )

        # Final summary
        total_audio_duration = len(watermarked_result) / self.sr
        print_generation_complete(total_time, total_audio_duration, num_chunks)

        return torch.from_numpy(watermarked_result).unsqueeze(0)

    def _split_text_intelligently(
        self, text: str, target_words_per_chunk: int = 50, language: str = "en"
    ) -> List[str]:
        """
        Split text at sentence boundaries using spacy, then group into chunks.
        Uses shared utility function for consistency across all TTS implementations.
        """
        return split_text_intelligently(text, target_words_per_chunk, language)

    def _crossfade_chunks(
        self, chunks: List[torch.Tensor], overlap_duration: float = 0.1
    ) -> torch.Tensor:
        """
        Concatenate audio chunks with crossfading.
        Uses shared utility function for consistency across all TTS implementations.
        """
        return crossfade_chunks(chunks, self.sr, overlap_duration)

    def generate_long(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        overlap_duration: float = 0.05,
        language: str = "en",
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.8,
        top_p: float = 1.0,
        min_p: float = 0.05,
        repetition_penalty: float = 1.2,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        Generate long-form speech with adaptive chunking strategy.

        Automatically chooses the best chunking strategy based on text length:
        - Short texts (< 50 words): Individual sentence processing (MLX optimal)
        - Long texts (>= 50 words): Grouped sentence processing (reduces overhead)

        This balances MLX's advantage on shorter sequences with the overhead
        of per-sentence generation for longer texts.

        Args:
            text: Input text to synthesize
            audio_prompt_path: Optional path to reference audio
            exaggeration: Emotion exaggeration factor
            cfg_weight: Classifier-free guidance weight
            overlap_duration: Crossfade duration in seconds between sentences
            language: Language code for spacy sentence tokenization (e.g., "en", "de", "fr")
            max_new_tokens: Maximum tokens to generate per sentence
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            min_p: Minimum probability threshold
            repetition_penalty: Repetition penalty factor

        Returns:
            Generated audio waveform as torch tensor
        """
        # Prepare conditioning first (only once)
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert (
                self.conds is not None
            ), "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Split text into individual sentences using spacy
        if SPACY_AVAILABLE:
            sentences = split_into_sentences(text, lang=language)
        else:
            # Fallback to regex
            sentence_pattern = r"(?<=[.!?])\s+"
            sentences = [
                s.strip() for s in re.split(sentence_pattern, text) if s.strip()
            ]

        if not sentences:
            sentences = [text]

        # Adaptive chunking: decide strategy based on total word count
        total_words = len(text.split())

        if total_words < ADAPTIVE_THRESHOLD_WORDS:
            # Short text: process each sentence individually (MLX optimal)
            chunks_to_generate = sentences
            chunking_strategy = "per-sentence"
        else:
            # Long text: group sentences to reduce per-chunk overhead
            chunks_to_generate = split_text_intelligently(
                text, TARGET_WORDS_PER_CHUNK, language
            )
            chunking_strategy = f"grouped (~{TARGET_WORDS_PER_CHUNK} words/chunk)"

        # Print chunk overview
        num_chunks = len(chunks_to_generate)
        print_generation_plan(
            total_words, chunks_to_generate, chunking_strategy, is_long_form=True
        )

        # Generate each chunk with status updates
        audio_chunks = []
        import time as _time

        total_start = _time.time()

        for i, chunk_text in enumerate(chunks_to_generate):
            print_chunk_generating(i, num_chunks, chunk_text)

            chunk_start = _time.time()

            # Generate without watermark for intermediate chunks
            chunk_audio = self._generate_single_sentence(
                chunk_text,
                cfg_weight=cfg_weight,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                apply_watermark=False,  # Apply watermark only to final result
                show_progress=True,  # Show tqdm progress for each chunk
            )

            chunk_time = _time.time() - chunk_start
            chunk_duration = (
                chunk_audio.shape[-1] / self.sr
                if hasattr(chunk_audio, "shape")
                else len(chunk_audio) / self.sr
            )

            print_chunk_completed(i, num_chunks, chunk_time, chunk_duration)

            audio_chunks.append(chunk_audio)

            # Memory cleanup between chunks
            import gc

            gc.collect()

            # Clear PyTorch MPS cache periodically (every 3 chunks)
            if i > 0 and i % 3 == 0 and torch.backends.mps.is_available():
                torch.mps.empty_cache()

        total_time = _time.time() - total_start

        # Crossfade chunks together
        print_crossfading(num_chunks)
        result = self._crossfade_chunks(audio_chunks, overlap_duration)

        # Apply watermark to final concatenated audio
        result_np = result.numpy() if isinstance(result, torch.Tensor) else result
        watermarked_result = self.watermarker.apply_watermark(
            result_np, sample_rate=self.sr
        )

        # Final summary
        total_audio_duration = len(watermarked_result) / self.sr
        print_generation_complete(total_time, total_audio_duration, num_chunks)

        return torch.from_numpy(watermarked_result).unsqueeze(0)


class ChatterboxTTSPureMLX:
    """
    Pure MLX Chatterbox TTS pipeline.

    Uses MLX for both T3 and S3Gen for maximum Apple Silicon optimization.
    - T3: MLX (autoregressive token generation)
    - S3Gen: MLX (flow matching + HiFiGAN vocoder)

    This provides the best performance on Apple Silicon but requires
    all operations to be implemented in MLX.
    """

    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3MLX,
        s3gen_mlx,  # S3Token2WavMLX
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str = "cpu",  # For VE only
        conds=None,
        ckpt_dir: Optional[Path] = None,
    ):
        """
        Initialize pure MLX TTS pipeline.

        Args:
            t3: T3MLX model (MLX)
            s3gen_mlx: S3Token2WavMLX model (MLX)
            ve: Voice encoder (PyTorch - used for conditioning)
            tokenizer: Text tokenizer
            device: PyTorch device for voice encoder
            conds: Optional pre-computed conditioning
            ckpt_dir: Path to checkpoint directory (for loading PyTorch S3Gen when needed)
        """
        self.sr = S3GEN_SR
        self.t3 = t3
        self.s3gen = s3gen_mlx
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self._ckpt_dir = ckpt_dir
        self._pt_s3gen = None  # Lazy-loaded PyTorch S3Gen for conditioning
        self.watermarker = perth.PerthImplicitWatermarker()

        # Conditioning cache for MLX conversion (avoids re-converting each sentence)
        self._cached_t3_cond_mx: Optional[T3CondMLX] = None
        self._cached_cond_hash: Optional[int] = None

    def _get_cached_t3_cond_mx(self) -> T3CondMLX:
        """Get or create cached MLX conditioning for T3.

        Caches the MLX conversion of T3 conditioning to avoid redundant
        PyTorch->MLX conversion overhead when generating multiple sentences.
        """
        if self.conds is None or self.conds.t3 is None:
            raise RuntimeError("No conditioning set. Call set_cond() first.")

        # Compute hash of current conditioning
        cond_hash = hash(
            (
                (
                    id(self.conds.t3.speaker_emb)
                    if self.conds.t3.speaker_emb is not None
                    else None
                ),
                (
                    id(self.conds.t3.cond_prompt_speech_tokens)
                    if self.conds.t3.cond_prompt_speech_tokens is not None
                    else None
                ),
                float(self.conds.t3.emotion_adv[0, 0, 0].item()),
            )
        )

        # Return cached if still valid
        if self._cached_t3_cond_mx is not None and self._cached_cond_hash == cond_hash:
            return self._cached_t3_cond_mx

        # Convert and cache
        self._cached_t3_cond_mx = T3CondMLX(
            speaker_emb=(
                mx.array(self.conds.t3.speaker_emb.cpu().numpy())
                if self.conds.t3.speaker_emb is not None
                else None
            ),
            cond_prompt_speech_tokens=(
                mx.array(self.conds.t3.cond_prompt_speech_tokens.cpu().numpy())
                if self.conds.t3.cond_prompt_speech_tokens is not None
                else None
            ),
            emotion_adv=float(self.conds.t3.emotion_adv[0, 0, 0].item()),
        )
        self._cached_cond_hash = cond_hash

        return self._cached_t3_cond_mx

    @classmethod
    def from_pretrained(
        cls,
        cache_dir: Optional[str] = None,
        t3_config: Optional[T3Config] = None,
        use_default_speaker: bool = True,
    ) -> "ChatterboxTTSPureMLX":
        """
        Load pre-trained Chatterbox TTS model with pure MLX.

        Args:
            cache_dir: Optional cache directory for model files
            t3_config: Optional T3 configuration (default: English-only)
            use_default_speaker: Whether to load default speaker conditioning

        Returns:
            ChatterboxTTSPureMLX instance
        """
        logger.info("Loading Chatterbox TTS with pure MLX optimization...")

        # Download model files from HuggingFace
        for fpath in [
            "ve.safetensors",
            "t3_cfg.safetensors",
            "s3gen.safetensors",
            "tokenizer.json",
            "conds.pt",
        ]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        ckpt_dir = Path(local_path).parent

        return cls.from_local(ckpt_dir, t3_config, use_default_speaker)

    @classmethod
    def from_local(
        cls,
        ckpt_dir: Path,
        t3_config: Optional[T3Config] = None,
        use_default_speaker: bool = True,
    ) -> "ChatterboxTTSPureMLX":
        """
        Load model from local checkpoint directory using pure MLX.

        Args:
            ckpt_dir: Path to checkpoint directory
            t3_config: Optional T3 configuration
            use_default_speaker: Whether to load default speaker

        Returns:
            ChatterboxTTSPureMLX instance
        """
        from .models.s3gen_mlx.s3gen_mlx import S3Token2WavMLX
        from .models.s3gen_mlx.convert_weights import convert_and_load_weights

        ckpt_dir = Path(ckpt_dir)
        device = "cpu"  # VE on CPU for compatibility

        # Load voice encoder (PyTorch - needed for conditioning)
        logger.info("Loading voice encoder...")
        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve.to(device).eval()

        # Initialize T3 MLX model
        logger.info("Initializing T3 MLX model...")
        if t3_config is None:
            t3_config = T3Config.english_only()

        t3 = T3MLX(hp=t3_config)

        # Load T3 weights
        logger.info("Loading T3 weights into MLX...")
        t3_ckpt = ckpt_dir / "t3_cfg.safetensors"
        pt_state = load_file(t3_ckpt)
        if "model" in pt_state.keys():
            pt_state = pt_state["model"][0]

        mlx_state = {}
        for k, v in pt_state.items():
            mlx_state[k] = mx.array(v.cpu().numpy())

        t3.load_weights(list(mlx_state.items()), strict=False)
        t3.eval()  # Set to eval mode to disable dropout
        logger.info("✓ T3 weights loaded successfully")

        # Load S3Gen MLX
        logger.info("Loading S3Gen MLX vocoder...")
        s3gen = S3Token2WavMLX()
        s3gen_weights = convert_and_load_weights(str(ckpt_dir / "s3gen.safetensors"))
        s3gen_weights = {k: mx.array(v) for k, v in s3gen_weights.items()}
        s3gen.load_weights(list(s3gen_weights.items()), strict=False)
        s3gen.eval()  # Set to eval mode
        logger.info("✓ S3Gen MLX loaded successfully")

        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = EnTokenizer(str(ckpt_dir / "tokenizer.json"))

        # Load default conditioning if requested
        conds = None
        if use_default_speaker:
            logger.info("Loading default speaker conditioning...")
            cond_path = ckpt_dir / "conds.pt"
            if cond_path.exists():
                from .tts import Conditionals

                conds = Conditionals.load(cond_path, map_location=device)
                logger.info("✓ Default speaker conditioning loaded")

        logger.info("✅ ChatterboxTTSPureMLX loaded successfully!")
        logger.info("   T3: MLX (Apple Silicon optimized)")
        logger.info("   S3Gen: MLX (Apple Silicon optimized)")

        return cls(
            t3=t3,
            s3gen_mlx=s3gen,
            ve=ve,
            tokenizer=tokenizer,
            device=device,
            conds=conds,
            ckpt_dir=ckpt_dir,
        )

    def _get_pt_s3gen(self):
        """Lazy-load PyTorch S3Gen for conditioning extraction."""
        if self._pt_s3gen is None:
            if self._ckpt_dir is None:
                raise ValueError("Cannot load PyTorch S3Gen: ckpt_dir not set")
            from .models.s3gen import S3Gen

            logger.info("Loading PyTorch S3Gen for conditioning...")
            self._pt_s3gen = S3Gen()
            self._pt_s3gen.load_state_dict(
                load_file(self._ckpt_dir / "s3gen.safetensors"), strict=False
            )
            self._pt_s3gen.to(self.device).eval()
        return self._pt_s3gen

    def prepare_conditionals(self, wav_fpath: str, exaggeration: float = 0.5):
        """
        Prepare conditioning from a reference audio file.

        Uses PyTorch S3Gen's embed_ref for conditioning extraction (same as hybrid),
        then converts to MLX arrays for Pure MLX inference pipeline.

        Args:
            wav_fpath: Path to reference audio file
            exaggeration: Emotion exaggeration factor (0.0 to 1.0)
        """
        from .tts import Conditionals

        # Load reference wav at S3GEN_SR (24kHz)
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        # Truncate to max conditioning length
        s3gen_ref_wav = s3gen_ref_wav[: self.DEC_COND_LEN]

        # Use loaded PyTorch S3Gen to compute reference embedding (consistent with hybrid)
        # This ensures mel, tokens, and speaker embedding are computed exactly the same way
        pt_s3gen = self._get_pt_s3gen()
        with torch.inference_mode():
            pt_ref_dict = pt_s3gen.embed_ref(
                s3gen_ref_wav, S3GEN_SR, device=self.device
            )

        # Convert PyTorch tensors to MLX arrays
        ref_mel_mx = mx.array(pt_ref_dict["prompt_feat"].cpu().numpy())  # [B, T, 80]
        ref_speech_tokens_mx = mx.array(
            pt_ref_dict["prompt_token"].cpu().numpy()
        )  # [B, T]
        spk_embed_mx = mx.array(pt_ref_dict["embedding"].cpu().numpy())  # [B, 192]

        # Get VE embedding for T3 (256 dims) - same as hybrid
        ve_embed = torch.from_numpy(
            self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR)
        )
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        # Speech tokens for T3 conditioning (truncated to prompt length)
        t3_cond_prompt_tokens = None
        if hasattr(self.t3, "hp") and self.t3.hp.speech_cond_prompt_len:
            plen = self.t3.hp.speech_cond_prompt_len
            s3_tokzr = pt_s3gen.tokenizer
            t3_tokens, _ = s3_tokzr.forward(
                [ref_16k_wav[: self.ENC_COND_LEN]], max_len=plen
            )
            t3_cond_prompt_tokens = torch.atleast_2d(t3_tokens).to(self.device)

        # Create reference dict for S3Gen MLX with MLX arrays
        s3gen_ref_dict = {
            "prompt_token": ref_speech_tokens_mx,  # [B, T] speech tokens from reference
            "prompt_token_len": int(pt_ref_dict["prompt_token_len"][0].item()),
            "prompt_feat": ref_mel_mx,  # [B, T, 80] mel features
            "prompt_feat_len": ref_mel_mx.shape[1],
            "embedding": spk_embed_mx,  # [B, 192] speaker embedding
        }

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)

        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

        # Clear conditioning cache when conditioning changes
        self._cached_t3_cond_mx = None
        self._cached_cond_hash = None
        _log_memory_mlx("pure_mlx_conditionals_prepared")

    def _generate_single_sentence(
        self,
        text: str,
        cfg_weight: float = 0.5,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.8,
        top_p: float = 1.0,
        min_p: float = 0.05,
        repetition_penalty: float = 1.2,
        apply_watermark: bool = True,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate speech for a single sentence/chunk (internal method).

        Assumes conditioning is already prepared. This is the core generation
        logic extracted for use by both generate() and generate_long().

        Args:
            text: Input text (single sentence/chunk)
            cfg_weight: Classifier-free guidance weight
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            min_p: Minimum probability threshold
            repetition_penalty: Repetition penalty factor
            apply_watermark: Whether to apply watermark (default True, False for intermediate chunks)
            show_progress: Whether to show token-level progress bar

        Returns:
            Generated audio waveform as torch tensor
        """
        # Normalize and tokenize text
        text = punc_norm(text)

        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        # Estimate reasonable max_new_tokens based on text length if not provided
        # Speech tokens are roughly 10-15x text characters, with 2x safety buffer
        # This prevents runaway generation when EOS is not triggered
        if max_new_tokens is None:
            estimated_tokens = len(text) * 15  # ~15 speech tokens per character
            max_new_tokens = min(
                max(estimated_tokens * 2, 200),  # At least 200, 2x buffer
                self.t3.hp.max_speech_tokens,  # But never exceed model max
            )
            logger.debug(
                f"[PureMLX] Estimated max_new_tokens: {max_new_tokens} for {len(text)} chars"
            )

        # Add start/end tokens for CFG
        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        # Convert to MLX for T3 inference
        text_tokens_mx = mx.array(text_tokens.cpu().numpy())

        # Get cached T3 conditioning (avoids re-conversion for each sentence)
        t3_cond_mx = self._get_cached_t3_cond_mx()

        _log_memory_mlx("pure_mlx_before_t3_inference")

        # Generate speech tokens with T3 MLX
        with torch.inference_mode():
            speech_tokens_mx = self.t3.inference(
                t3_cond=t3_cond_mx,
                text_tokens=text_tokens_mx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                show_progress=show_progress,
            )

        # Clear T3's KV cache to free memory after generation
        if hasattr(self.t3, "patched_model") and self.t3.patched_model is not None:
            self.t3.patched_model.reset_state()

        _log_memory_mlx("pure_mlx_after_t3_inference")

        # Extract conditional batch (first one)
        speech_tokens = (
            speech_tokens_mx[0] if len(speech_tokens_mx.shape) > 1 else speech_tokens_mx
        )

        # Clean up tokens
        speech_tokens_np = np.array(speech_tokens).astype(np.int64)
        # Filter out special tokens (>= 6561)
        valid_mask = speech_tokens_np < 6561
        speech_tokens_np = speech_tokens_np[valid_mask]

        # Convert to MLX array
        speech_tokens_mlx = mx.array(speech_tokens_np.reshape(1, -1))

        # Get prompt tokens from conditioning
        prompt_tokens = self.conds.gen.get("prompt_token")
        if prompt_tokens is None:
            # Use tokens from T3 cond
            if self.conds.t3.cond_prompt_speech_tokens is not None:
                prompt_tokens = mx.array(
                    self.conds.t3.cond_prompt_speech_tokens.cpu().numpy()
                )
            else:
                prompt_tokens = mx.zeros((1, 1), dtype=mx.int32)
        elif isinstance(prompt_tokens, torch.Tensor):
            prompt_tokens = mx.array(prompt_tokens.cpu().numpy())
        elif not isinstance(prompt_tokens, mx.array):
            prompt_tokens = mx.array(prompt_tokens)

        # Convert prompt_feat to MLX if it's a PyTorch tensor
        prompt_feat = self.conds.gen.get("prompt_feat")
        if isinstance(prompt_feat, torch.Tensor):
            prompt_feat = mx.array(prompt_feat.cpu().numpy())
        elif prompt_feat is not None and not isinstance(prompt_feat, mx.array):
            prompt_feat = mx.array(prompt_feat)

        # Convert embedding to MLX if it's a PyTorch tensor
        embedding = self.conds.gen.get("embedding")
        if isinstance(embedding, torch.Tensor):
            embedding = mx.array(embedding.detach().cpu().numpy())
        elif embedding is not None and not isinstance(embedding, mx.array):
            embedding = mx.array(embedding)

        # Prepare ref_dict for S3Gen MLX
        ref_dict = {
            "prompt_token": prompt_tokens,
            "prompt_token_len": (
                int(prompt_tokens.shape[1]) if hasattr(prompt_tokens, "shape") else 1
            ),
            "prompt_feat": prompt_feat,
            "prompt_feat_len": (
                int(prompt_feat.shape[1])
                if prompt_feat is not None and hasattr(prompt_feat, "shape")
                else 0
            ),
            "embedding": embedding,
        }

        _log_memory_mlx("pure_mlx_before_s3gen_inference")

        # Generate waveform with S3Gen MLX
        wav_mlx, _ = self.s3gen.inference(speech_tokens_mlx, ref_dict, finalize=True)
        wav = np.array(wav_mlx.squeeze(0))

        _log_memory_mlx("pure_mlx_after_s3gen_inference")

        if apply_watermark:
            wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)

        return torch.from_numpy(wav).unsqueeze(0)

    def generate(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.8,
        top_p: float = 1.0,
        min_p: float = 0.05,
        repetition_penalty: float = 1.2,
        use_sentence_chunking: bool = True,
        overlap_duration: float = 0.05,
        language: str = "en",
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate speech from text using pure MLX pipeline.

        By default, splits text into sentences using spacy for optimal performance.

        Args:
            text: Input text to synthesize
            audio_prompt_path: Optional path to reference audio for voice cloning
            exaggeration: Emotion exaggeration factor (0.0 to 1.0)
            cfg_weight: Classifier-free guidance weight
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            min_p: Minimum probability threshold
            repetition_penalty: Repetition penalty factor
            use_sentence_chunking: Whether to split text into sentences (default True for performance)
            overlap_duration: Crossfade duration between sentences in seconds
            language: Language code for spacy sentence tokenization (e.g., "en", "de", "fr")
            show_progress: Whether to show token-level progress bar (default True)

        Returns:
            Generated audio waveform as torch tensor
        """
        import time as _time

        # Prepare conditioning if audio prompt provided
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert (
                self.conds is not None
            ), "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Split text into sentences for optimal performance
        if use_sentence_chunking and SPACY_AVAILABLE:
            sentences = split_into_sentences(text, lang=language)
        else:
            sentences = [text]

        total_words = len(text.split())
        num_chunks = len(sentences)

        # Generate audio for each sentence
        if len(sentences) == 1:
            # Single sentence - generate directly with watermark
            print_generation_plan(
                total_words, sentences, "single chunk", prefix="[PureMLX] "
            )
            print_chunk_generating(0, 1, sentences[0])
            gen_start = _time.time()

            result = self._generate_single_sentence(
                sentences[0],
                cfg_weight=cfg_weight,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                apply_watermark=True,
                show_progress=show_progress,  # Use caller's preference
            )

            gen_time = _time.time() - gen_start
            audio_duration = (
                result.shape[-1] / self.sr
                if hasattr(result, "shape")
                else len(result) / self.sr
            )
            print_chunk_completed(0, 1, gen_time, audio_duration)
            print_generation_complete(gen_time, audio_duration, prefix="[PureMLX] ")

            return result

        # Multiple sentences - print overview and generate each with status updates
        print_generation_plan(
            total_words, sentences, "per-sentence", prefix="[PureMLX] "
        )

        # Generate each chunk with status updates
        audio_chunks = []
        total_start = _time.time()

        for i, sentence in enumerate(sentences):
            print_chunk_generating(i, num_chunks, sentence)
            chunk_start = _time.time()

            # Don't apply watermark to intermediate chunks
            chunk_audio = self._generate_single_sentence(
                sentence,
                cfg_weight=cfg_weight,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                apply_watermark=False,
                show_progress=True,  # Show tqdm progress for each chunk
            )

            chunk_time = _time.time() - chunk_start
            chunk_duration = (
                chunk_audio.shape[-1] / self.sr
                if hasattr(chunk_audio, "shape")
                else len(chunk_audio) / self.sr
            )
            print_chunk_completed(i, num_chunks, chunk_time, chunk_duration)

            audio_chunks.append(chunk_audio)

            # Memory cleanup between chunks
            # Note: mx.clear_cache() is already called in T3's generate() and reset_state()
            import gc

            gc.collect()

            # Clear PyTorch MPS cache periodically (every 3 chunks)
            if i > 0 and i % 3 == 0 and torch.backends.mps.is_available():
                torch.mps.empty_cache()

        total_time = _time.time() - total_start

        # Crossfade chunks together
        print_crossfading(num_chunks)
        result = self._crossfade_chunks(audio_chunks, overlap_duration)

        # Apply watermark to final concatenated audio
        result_np = result.numpy() if isinstance(result, torch.Tensor) else result
        watermarked_result = self.watermarker.apply_watermark(
            result_np, sample_rate=self.sr
        )

        # Final summary
        total_audio_duration = len(watermarked_result) / self.sr
        print_generation_complete(
            total_time, total_audio_duration, num_chunks, prefix="[PureMLX] "
        )

        return torch.from_numpy(watermarked_result).unsqueeze(0)

    def _split_text_intelligently(
        self, text: str, target_words_per_chunk: int = 50, language: str = "en"
    ) -> List[str]:
        """
        Split text at sentence boundaries using spacy, then group into chunks.
        Uses shared utility function for consistency across all TTS implementations.
        """
        return split_text_intelligently(text, target_words_per_chunk, language)

    def _crossfade_chunks(
        self, chunks: List[torch.Tensor], overlap_duration: float = 0.1
    ) -> torch.Tensor:
        """
        Concatenate audio chunks with crossfading.
        Uses shared utility function for consistency across all TTS implementations.
        """
        return crossfade_chunks(chunks, self.sr, overlap_duration)

    def generate_long(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        overlap_duration: float = 0.05,
        language: str = "en",
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.8,
        top_p: float = 1.0,
        min_p: float = 0.05,
        repetition_penalty: float = 1.2,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        Generate long-form speech with adaptive chunking strategy.

        Automatically chooses the best chunking strategy based on text length:
        - Short texts (< 50 words): Individual sentence processing (MLX optimal)
        - Long texts (>= 50 words): Grouped sentence processing (reduces overhead)

        This balances MLX's advantage on shorter sequences with the overhead
        of per-sentence generation for longer texts.

        Args:
            text: Input text to synthesize
            audio_prompt_path: Optional path to reference audio
            exaggeration: Emotion exaggeration factor
            cfg_weight: Classifier-free guidance weight
            overlap_duration: Crossfade duration in seconds between sentences
            language: Language code for spacy sentence tokenization (e.g., "en", "de", "fr")
            max_new_tokens: Maximum tokens to generate per sentence
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            min_p: Minimum probability threshold
            repetition_penalty: Repetition penalty factor

        Returns:
            Generated audio waveform as torch tensor
        """
        # Prepare conditioning first (only once)
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert (
                self.conds is not None
            ), "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Split text into individual sentences using spacy
        if SPACY_AVAILABLE:
            sentences = split_into_sentences(text, lang=language)
        else:
            # Fallback to regex
            sentence_pattern = r"(?<=[.!?])\s+"
            sentences = [
                s.strip() for s in re.split(sentence_pattern, text) if s.strip()
            ]

        if not sentences:
            sentences = [text]

        # Adaptive chunking: decide strategy based on total word count
        total_words = len(text.split())

        if total_words < ADAPTIVE_THRESHOLD_WORDS:
            # Short text: process each sentence individually (MLX optimal)
            chunks_to_generate = sentences
            chunking_strategy = "per-sentence"
        else:
            # Long text: group sentences to reduce per-chunk overhead
            chunks_to_generate = split_text_intelligently(
                text, TARGET_WORDS_PER_CHUNK, language
            )
            chunking_strategy = f"grouped (~{TARGET_WORDS_PER_CHUNK} words/chunk)"

        # Print chunk overview
        num_chunks = len(chunks_to_generate)
        print_generation_plan(
            total_words,
            chunks_to_generate,
            chunking_strategy,
            is_long_form=True,
            prefix="[PureMLX] ",
        )

        # Generate each chunk with status updates
        audio_chunks = []
        import time as _time

        total_start = _time.time()

        for i, chunk_text in enumerate(chunks_to_generate):
            print_chunk_generating(i, num_chunks, chunk_text)

            chunk_start = _time.time()

            # Generate without watermark for intermediate chunks
            chunk_audio = self._generate_single_sentence(
                chunk_text,
                cfg_weight=cfg_weight,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                apply_watermark=False,  # Apply watermark only to final result
                show_progress=True,  # Show tqdm progress for each chunk
            )

            chunk_time = _time.time() - chunk_start
            chunk_duration = (
                chunk_audio.shape[-1] / self.sr
                if hasattr(chunk_audio, "shape")
                else len(chunk_audio) / self.sr
            )

            print_chunk_completed(i, num_chunks, chunk_time, chunk_duration)

            audio_chunks.append(chunk_audio)

            # Memory cleanup between chunks
            import gc

            gc.collect()

            # Clear PyTorch MPS cache periodically (every 3 chunks)
            if i > 0 and i % 3 == 0 and torch.backends.mps.is_available():
                torch.mps.empty_cache()

        total_time = _time.time() - total_start

        # Crossfade chunks together
        print_crossfading(num_chunks)
        result = self._crossfade_chunks(audio_chunks, overlap_duration)

        # Apply watermark to final concatenated audio
        result_np = result.numpy() if isinstance(result, torch.Tensor) else result
        watermarked_result = self.watermarker.apply_watermark(
            result_np, sample_rate=self.sr
        )

        # Final summary
        total_audio_duration = len(watermarked_result) / self.sr
        print_generation_complete(
            total_time, total_audio_duration, num_chunks, prefix="[PureMLX] "
        )

        return torch.from_numpy(watermarked_result).unsqueeze(0)
