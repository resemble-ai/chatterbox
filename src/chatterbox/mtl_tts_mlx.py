# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
MLX-optimized Multilingual TTS pipeline for Chatterbox.
Provides significant performance improvements on Apple Silicon (M1/M2/M3/M4).

This implementation uses a hybrid approach:
- MLX for T3 (autoregressive text-to-speech token generation) - main performance win
- PyTorch for S3Gen (vocoder) - uses MPS acceleration on Apple Silicon

Supports 23 languages including English, Spanish, French, German, Japanese, Chinese, etc.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict
import logging
import re
import os

import numpy as np

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
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors

from .models.t3_mlx.t3_mlx import T3MLX
from .models.t3_mlx.modules.cond_enc_mlx import T3CondMLX
from .models.t3.modules.t3_config import T3Config
from .models.t3.modules.cond_enc import T3Cond

# Use PyTorch S3Gen for now (hybrid approach - T3 in MLX, S3Gen in PyTorch/MPS)
from .models.s3gen import S3Gen, S3GEN_SR
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.voice_encoder import VoiceEncoder
from .models.tokenizers import MTLTokenizer
from .models.utils import clear_device_memory

# Shared generation utilities
from .generation_utils import (
    SPACY_AVAILABLE,
    split_into_sentences,
    get_adaptive_chunks,
    split_text_intelligently,
    crossfade_chunks,
    estimate_max_tokens,
    print_generation_plan,
    print_chunk_generating,
    print_chunk_completed,
    print_generation_complete,
    print_crossfading,
)

import perth
import librosa

logger = logging.getLogger(__name__)

REPO_ID = "ResembleAI/chatterbox"

# Supported languages for the multilingual model
SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}


def punc_norm(text: str, debug: bool = True) -> str:
    """
    Quick cleanup func for punctuation from LLMs or
    containing chars not seen often in the dataset
    """
    original_text = text

    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        (
            """, '"'),
        (""",
            '"',
        ),
        ("'", "'"),
        ("'", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Clean up leading quotes and spaces
    # This handles sentences that start with: '" "No entendía...' -> 'No entendía...'
    while text and text[0] in "\"'":
        text = text[1:].lstrip()

    # Clean up trailing quotes and spaces in various patterns
    # Remove trailing whitespace first
    text = text.rstrip()

    # Remove standalone quotes at end (with or without spaces before them)
    # This handles: '." "' -> '."' and then '."' -> '.'
    while text and text[-1] in "\"'":
        text = text[:-1].rstrip()

    # Now handle punctuation followed by quotes: '."' -> '.'
    text = re.sub(r'([.!?,])["\']+$', r"\1", text)
    text = re.sub(r'["\']+([.!?,])$', r"\1", text)

    # Clean up double punctuation (e.g., ".." or ",," but not "...")
    text = re.sub(r"([.!?,])\1+", r"\1", text)

    # Clean up double spaces that may have been introduced
    text = " ".join(text.split())

    # Skip empty text after cleanup
    if len(text) == 0:
        return ""

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",", "、", "，", "。", "？", "！"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    # DEBUG: Log punc_norm changes
    if debug and original_text != text:
        logger.info(f"[punc_norm DEBUG] Input:  {repr(original_text[:200])}")
        logger.info(f"[punc_norm DEBUG] Output: {repr(text[:200])}")

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen (same structure as mtl_tts.py)
    """

    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(t3=self.t3.__dict__, gen=self.gen)
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs["t3"]), kwargs["gen"])


class ChatterboxMultilingualTTSMLX:
    """
    MLX-optimized Multilingual Chatterbox TTS pipeline.

    Uses a hybrid approach for best performance on Apple Silicon:
    - MLX for T3 (autoregressive token generation) - main speedup
    - PyTorch/MPS for S3Gen (vocoder) - fast on Apple Silicon

    Supports 23 languages including English, Spanish, French, German,
    Japanese, Chinese, Korean, Arabic, Hindi, etc.
    """

    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3MLX,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: MTLTokenizer,
        device: str = "mps",
        conds: Conditionals = None,
    ):
        """
        Initialize hybrid MLX/PyTorch multilingual TTS pipeline.

        Args:
            t3: T3MLX model (MLX)
            s3gen: S3Gen vocoder (PyTorch)
            ve: Voice encoder (PyTorch)
            tokenizer: Multilingual text tokenizer
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

    @classmethod
    def get_supported_languages(cls) -> Dict[str, str]:
        """Return dictionary of supported language codes and names."""
        return SUPPORTED_LANGUAGES.copy()

    @classmethod
    def from_pretrained(
        cls,
        cache_dir: Optional[str] = None,
        t3_config: Optional[T3Config] = None,
        use_default_speaker: bool = True,
        device: str = "mps",
    ) -> "ChatterboxMultilingualTTSMLX":
        """
        Load pre-trained multilingual Chatterbox TTS model with MLX optimization.

        Args:
            cache_dir: Optional cache directory for model files
            t3_config: Optional T3 configuration (default: multilingual)
            use_default_speaker: Whether to load default speaker conditioning
            device: PyTorch device for S3Gen ("mps" or "cpu")

        Returns:
            ChatterboxMultilingualTTSMLX instance
        """
        logger.info("Loading Chatterbox Multilingual TTS with MLX optimization...")

        # Download model files from HuggingFace
        ckpt_dir = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main",
                allow_patterns=[
                    "ve.pt",
                    "t3_mtl23ls_v2.safetensors",
                    "s3gen.pt",
                    "grapheme_mtl_merged_expanded_v1.json",
                    "conds.pt",
                    "Cangjie5_TC.json",
                ],
                token=os.getenv("HF_TOKEN"),
            )
        )

        return cls.from_local(ckpt_dir, t3_config, use_default_speaker, device)

    @classmethod
    def from_local(
        cls,
        ckpt_dir: Path,
        t3_config: Optional[T3Config] = None,
        use_default_speaker: bool = True,
        device: str = "mps",
    ) -> "ChatterboxMultilingualTTSMLX":
        """
        Load model from local checkpoint directory.

        Args:
            ckpt_dir: Path to checkpoint directory
            t3_config: Optional T3 configuration
            use_default_speaker: Whether to load default speaker
            device: PyTorch device for S3Gen

        Returns:
            ChatterboxMultilingualTTSMLX instance
        """
        ckpt_dir = Path(ckpt_dir)

        # Check MPS availability
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS not available, falling back to CPU")
            device = "cpu"

        # Always load to CPU first for non-CUDA devices
        map_location = torch.device("cpu") if device in ["cpu", "mps"] else None

        # Load voice encoder (PyTorch)
        logger.info("Loading voice encoder...")
        ve = VoiceEncoder()
        ve.load_state_dict(
            torch.load(ckpt_dir / "ve.pt", weights_only=True, map_location=map_location)
        )
        ve.to(device).eval()

        # Initialize T3 MLX model with multilingual config
        logger.info("Initializing T3 MLX model (multilingual)...")
        if t3_config is None:
            t3_config = T3Config.multilingual()

        t3 = T3MLX(hp=t3_config)

        # Load T3 multilingual weights and convert to MLX
        logger.info("Loading T3 multilingual weights into MLX...")
        t3_ckpt = ckpt_dir / "t3_mtl23ls_v2.safetensors"
        pt_state = load_safetensors(t3_ckpt)
        if "model" in pt_state.keys():
            pt_state = pt_state["model"][0]

        # Convert PyTorch weights to MLX format
        mlx_state = {}
        for k, v in pt_state.items():
            mlx_state[k] = mx.array(v.cpu().numpy())

        # Load weights into T3MLX with strict=False to ignore unused embed_tokens
        t3.load_weights(list(mlx_state.items()), strict=False)
        t3.eval()  # Set to eval mode to disable dropout
        logger.info("✓ T3 multilingual weights loaded successfully")

        # Load S3Gen (PyTorch/MPS) - hybrid approach
        logger.info(f"Loading S3Gen vocoder (PyTorch on {device})...")
        s3gen = S3Gen()
        s3gen.load_state_dict(
            torch.load(
                ckpt_dir / "s3gen.pt", weights_only=True, map_location=map_location
            )
        )
        s3gen.to(device).eval()
        logger.info("✓ S3Gen loaded successfully")

        # Load multilingual tokenizer
        logger.info("Loading multilingual tokenizer...")
        tokenizer = MTLTokenizer(str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"))

        # Load default conditioning if requested
        conds = None
        if use_default_speaker:
            logger.info("Loading default speaker conditioning...")
            cond_path = ckpt_dir / "conds.pt"
            if cond_path.exists():
                conds = Conditionals.load(cond_path, map_location=map_location).to(
                    device
                )
                logger.info("✓ Default speaker conditioning loaded")

        logger.info("✅ ChatterboxMultilingualTTSMLX loaded successfully!")
        logger.info("   T3: MLX (Apple Silicon optimized)")
        logger.info(f"   S3Gen: PyTorch on {device}")
        logger.info(f"   Supported languages: {len(SUPPORTED_LANGUAGES)}")

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
        # Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[: self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(
            s3gen_ref_wav, S3GEN_SR, device=self.device
        )

        # Speech cond prompt tokens
        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            # Limit audio length more aggressively for memory safety
            safe_audio_len = min(len(ref_16k_wav), 6 * S3_SR)
            limited_audio = ref_16k_wav[:safe_audio_len]

            # Memory cleanup before tokenization
            clear_device_memory()

            # Use smaller max_len to be extra safe
            safe_max_len = min(plen, 150)
            t3_cond_prompt_tokens, _ = s3_tokzr.forward(
                [limited_audio], max_len=safe_max_len
            )
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(
                self.device
            )

            # More memory cleanup after tokenization
            clear_device_memory()

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

    def generate(
        self,
        text: str,
        language_id: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 2.0,
        min_p: float = 0.05,
        top_p: float = 1.0,
        max_new_tokens: Optional[int] = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate speech from text using hybrid MLX/PyTorch pipeline.

        By default, splits text into sentences for optimal generation quality.

        Args:
            text: Input text to synthesize
            language_id: Language code (e.g., 'en', 'es', 'ja', 'zh')
            audio_prompt_path: Optional path to reference audio for voice cloning
            exaggeration: Emotion exaggeration factor (0.0 to 1.0)
            cfg_weight: Classifier-free guidance weight
            temperature: Sampling temperature
            repetition_penalty: Repetition penalty factor
            min_p: Minimum probability threshold
            top_p: Nucleus sampling threshold
            max_new_tokens: Maximum tokens to generate (auto-estimated if None)
            show_progress: Whether to show token-level progress bar (default True)

        Returns:
            Generated audio waveform as torch tensor
        """
        import time as _time

        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        # Prepare conditioning if audio prompt provided
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert (
                self.conds is not None
            ), "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
            _cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Normalize text
        text = punc_norm(text)

        # Split text into sentences for optimal generation
        lang = language_id.lower() if language_id else "en"
        if SPACY_AVAILABLE:
            sentences = split_into_sentences(text, lang=lang)
        else:
            sentences = [text]

        # Clean up each sentence (remove orphaned quotes from splitting)
        sentences = [punc_norm(s, debug=False) for s in sentences]
        # Filter out empty sentences
        sentences = [s for s in sentences if s and s.strip()]

        total_words = len(text.split())
        num_chunks = len(sentences)
        lang_name = SUPPORTED_LANGUAGES.get(lang, language_id)

        # Print generation plan
        print_generation_plan(
            total_words,
            sentences,
            "per-sentence",
            prefix=f"[MLX Multilingual - {lang_name}] ",
        )

        # Generate audio for each sentence
        if len(sentences) == 1:
            # Single sentence - generate directly
            print_chunk_generating(0, 1, sentences[0])
            gen_start = _time.time()

            result = self._generate_single(
                sentences[0],
                language_id=language_id,
                cfg_weight=cfg_weight,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                show_progress=show_progress,  # Use caller's preference
            )

            gen_time = _time.time() - gen_start
            audio_duration = (
                result.shape[-1] / self.sr
                if hasattr(result, "shape")
                else len(result) / self.sr
            )
            print_chunk_completed(0, 1, gen_time, audio_duration)
            print_generation_complete(
                gen_time, audio_duration, 1, prefix="[MLX Multilingual] "
            )

            clear_device_memory()
            return result

        # Multiple sentences - generate each and crossfade
        audio_chunks = []
        total_start = _time.time()

        for i, sentence in enumerate(sentences):
            len(sentence.split())
            print_chunk_generating(i, num_chunks, sentence)
            chunk_start = _time.time()

            chunk_audio = self._generate_single(
                sentence,
                language_id=language_id,
                cfg_weight=cfg_weight,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                show_progress=False,  # Suppress token progress, observability handles chunk progress
            )

            chunk_time = _time.time() - chunk_start
            chunk_duration = (
                chunk_audio.shape[-1] / self.sr
                if hasattr(chunk_audio, "shape")
                else len(chunk_audio) / self.sr
            )
            print_chunk_completed(i, num_chunks, chunk_time, chunk_duration)

            audio_chunks.append(chunk_audio)

        total_time = _time.time() - total_start

        # Crossfade chunks together
        print_crossfading(num_chunks)
        result = crossfade_chunks(audio_chunks, self.sr, 0.05)

        # Final summary
        result_np = result.numpy() if isinstance(result, torch.Tensor) else result
        total_audio_duration = len(result_np) / self.sr
        print_generation_complete(
            total_time, total_audio_duration, num_chunks, prefix="[MLX Multilingual] "
        )

        clear_device_memory()

        return (
            torch.from_numpy(result_np).unsqueeze(0)
            if isinstance(result_np, np.ndarray)
            else result.unsqueeze(0)
        )

    def _generate_single(
        self,
        text: str,
        language_id: str,
        cfg_weight: float = 0.5,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.8,
        repetition_penalty: float = 2.0,
        min_p: float = 0.05,
        top_p: float = 1.0,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate speech for a single sentence/chunk (internal method).

        Args:
            text: Input text (single sentence/chunk)
            language_id: Language code
            cfg_weight: Classifier-free guidance weight
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            repetition_penalty: Repetition penalty factor
            min_p: Minimum probability threshold
            top_p: Nucleus sampling threshold
            show_progress: Whether to show token-level progress bar

        Returns:
            Generated audio waveform as torch tensor
        """
        # Normalize and tokenize
        text = punc_norm(text)

        # Estimate max_new_tokens if not provided
        if max_new_tokens is None:
            max_new_tokens = estimate_max_tokens(text, self.t3.hp.max_speech_tokens)

        text_tokens = self.tokenizer.text_to_tokens(
            text, language_id=language_id.lower() if language_id else None
        ).to(self.device)

        # Add start/end tokens for CFG
        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        # Convert to MLX for T3 inference
        text_tokens_mx = mx.array(text_tokens.cpu().numpy())

        # Convert T3 conditioning to MLX
        t3_cond_mx = T3CondMLX(
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

        # Extract conditional batch (first one)
        speech_tokens = (
            speech_tokens_mx[0] if len(speech_tokens_mx.shape) > 1 else speech_tokens_mx
        )

        # Convert back to PyTorch for S3Gen
        speech_tokens_np = np.array(speech_tokens).astype(np.int64)
        speech_tokens_pt = torch.from_numpy(speech_tokens_np)

        # Drop invalid tokens (SOS/EOS)
        speech_tokens_pt = drop_invalid_tokens(speech_tokens_pt)

        # Filter out tokens >= 6561 (special tokens)
        speech_tokens_pt = speech_tokens_pt[speech_tokens_pt < 6561]
        speech_tokens_pt = speech_tokens_pt.to(self.device)

        # Generate waveform with S3Gen (PyTorch/MPS)
        with torch.inference_mode():
            wav, sources = self.s3gen.inference(
                speech_tokens=speech_tokens_pt,
                ref_dict=self.conds.gen,
            )
            if sources is not None:
                del sources
            wav = wav.squeeze(0).detach().cpu().numpy()

        return torch.from_numpy(wav).unsqueeze(0)

    def _split_text_intelligently(
        self, text: str, language_id: str, target_words_per_chunk: int = 50
    ) -> List[str]:
        """
        Split text at sentence/phrase boundaries for chunked generation.
        Uses shared utility with spacy support.
        """
        return split_text_intelligently(
            text,
            target_words_per_chunk,
            lang=language_id.lower() if language_id else "en",
        )

    def _crossfade_chunks(
        self, chunks: List[torch.Tensor], overlap_duration: float = 0.1
    ) -> np.ndarray:
        """
        Concatenate audio chunks with crossfading.
        Uses shared optimized utility.
        """
        result = crossfade_chunks(chunks, self.sr, overlap_duration)
        return result.numpy() if isinstance(result, torch.Tensor) else result

    def generate_long(
        self,
        text: str,
        language_id: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 2.0,
        min_p: float = 0.05,
        top_p: float = 1.0,
        overlap_duration: float = 0.05,
        max_new_tokens: Optional[int] = None,
        progress_callback=None,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        Generate long-form speech with adaptive chunking strategy.

        Automatically chooses the best chunking strategy based on text length:
        - Short texts (< 50 words): Individual sentence processing (MLX optimal)
        - Long texts (>= 50 words): Grouped sentence processing (reduces overhead)

        Args:
            text: Input text to synthesize (any length)
            language_id: Language code (e.g., 'en', 'es', 'ja', 'zh')
            audio_prompt_path: Path to reference audio file for voice cloning
            exaggeration: Voice exaggeration/expressiveness level (0.0-1.0)
            cfg_weight: Classifier-free guidance weight (0.0-1.0)
            temperature: Sampling temperature for token generation
            repetition_penalty: Penalty for repeating tokens
            min_p: Minimum probability threshold for sampling
            top_p: Nucleus sampling threshold
            overlap_duration: Duration in seconds of crossfade between chunks
            max_new_tokens: Maximum tokens to generate per chunk
            progress_callback: Optional callback function for progress monitoring

        Returns:
            torch.Tensor: Generated audio waveform with shape (1, num_samples)
        """
        import time as _time

        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        # Prepare initial conditioning
        if audio_prompt_path:
            if progress_callback:
                progress_callback(
                    stage="preparing_conditionals", audio_path=audio_prompt_path
                )
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert (
                self.conds is not None
            ), "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
            _cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Get adaptive chunks based on text length
        lang = language_id.lower() if language_id else "en"
        chunks_to_generate, chunking_strategy = get_adaptive_chunks(text, lang=lang)

        if not chunks_to_generate:
            chunks_to_generate = [text]

        total_words = len(text.split())
        num_chunks = len(chunks_to_generate)
        lang_name = SUPPORTED_LANGUAGES.get(lang, language_id)

        if progress_callback:
            progress_callback(
                stage="text_split",
                total_chunks=num_chunks,
                chunk_previews=[
                    (i + 1, len(chunk.split()), chunk[:50])
                    for i, chunk in enumerate(chunks_to_generate)
                ],
            )

        # Print generation plan
        print_generation_plan(
            total_words,
            chunks_to_generate,
            chunking_strategy,
            prefix=f"[MLX Multilingual - {lang_name}] ",
            is_long_form=True,
        )

        # Generate each chunk with status updates
        audio_chunks = []
        total_start = _time.time()

        for i, chunk_text in enumerate(chunks_to_generate):
            chunk_words = len(chunk_text.split())

            if progress_callback:
                progress_callback(
                    stage="chunk_start",
                    chunk_index=i,
                    chunk_number=i + 1,
                    total_chunks=num_chunks,
                    text_preview=chunk_text[:50],
                    word_count=chunk_words,
                )

            print_chunk_generating(i, num_chunks, chunk_text)
            chunk_start = _time.time()

            # Generate chunk
            chunk_audio = self._generate_single(
                chunk_text,
                language_id=language_id,
                cfg_weight=cfg_weight,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                show_progress=False,  # Suppress token progress, observability handles chunk progress
            )

            chunk_time = _time.time() - chunk_start
            chunk_duration = (
                chunk_audio.shape[-1] / self.sr
                if hasattr(chunk_audio, "shape")
                else len(chunk_audio) / self.sr
            )
            print_chunk_completed(i, num_chunks, chunk_time, chunk_duration)

            # MEMORY OPTIMIZATION: Move to CPU immediately
            if isinstance(chunk_audio, torch.Tensor):
                chunk_audio = chunk_audio.detach().cpu()
            audio_chunks.append(chunk_audio)

            # Aggressive memory cleanup after each chunk
            clear_device_memory()

            if progress_callback:
                progress_callback(
                    stage="chunk_complete",
                    chunk_index=i,
                    chunk_number=i + 1,
                    total_chunks=num_chunks,
                    audio_shape=(
                        chunk_audio.shape
                        if hasattr(chunk_audio, "shape")
                        else (len(chunk_audio),)
                    ),
                )

        total_time = _time.time() - total_start

        # Crossfade and concatenate all chunks
        if progress_callback:
            progress_callback(
                stage="crossfading",
                total_chunks=len(audio_chunks),
                overlap_duration=overlap_duration,
            )

        print_crossfading(num_chunks)
        result = crossfade_chunks(audio_chunks, self.sr, overlap_duration)

        # Final summary
        result_np = result.numpy() if isinstance(result, torch.Tensor) else result
        total_audio_duration = len(result_np) / self.sr
        print_generation_complete(
            total_time, total_audio_duration, num_chunks, prefix="[MLX Multilingual] "
        )

        if progress_callback:
            progress_callback(
                stage="complete",
                total_chunks=len(audio_chunks),
                final_audio_shape=(1, len(result_np)),
            )

        return (
            torch.from_numpy(result_np).unsqueeze(0)
            if isinstance(result_np, np.ndarray)
            else result.unsqueeze(0)
        )
