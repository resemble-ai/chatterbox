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
    import mlx.nn as nn
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


def punc_norm(text: str) -> str:
    """
    Quick cleanup func for punctuation from LLMs or
    containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

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
        (""", "\""),
        (""", "\""),
        ("'", "'"),
        ("'", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",", "、", "，", "。", "？", "！"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

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
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


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
    ) -> 'ChatterboxMultilingualTTSMLX':
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
    ) -> 'ChatterboxMultilingualTTSMLX':
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
        map_location = torch.device('cpu') if device in ["cpu", "mps"] else None

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
            torch.load(ckpt_dir / "s3gen.pt", weights_only=True, map_location=map_location)
        )
        s3gen.to(device).eval()
        logger.info("✓ S3Gen loaded successfully")

        # Load multilingual tokenizer
        logger.info("Loading multilingual tokenizer...")
        tokenizer = MTLTokenizer(
            str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json")
        )

        # Load default conditioning if requested
        conds = None
        if use_default_speaker:
            logger.info("Loading default speaker conditioning...")
            cond_path = ckpt_dir / "conds.pt"
            if cond_path.exists():
                conds = Conditionals.load(cond_path, map_location=map_location).to(device)
                logger.info("✓ Default speaker conditioning loaded")

        logger.info("✅ ChatterboxMultilingualTTSMLX loaded successfully!")
        logger.info(f"   T3: MLX (Apple Silicon optimized)")
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

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            # Limit audio length more aggressively for memory safety
            safe_audio_len = min(len(ref_16k_wav), 6 * S3_SR)
            limited_audio = ref_16k_wav[:safe_audio_len]

            # Memory cleanup before tokenization
            import gc
            clear_device_memory()

            # Use smaller max_len to be extra safe
            safe_max_len = min(plen, 150)
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([limited_audio], max_len=safe_max_len)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

            # More memory cleanup after tokenization
            clear_device_memory()

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
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
    ) -> torch.Tensor:
        """
        Generate speech from text using hybrid MLX/PyTorch pipeline.

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

        Returns:
            Generated audio waveform as torch tensor
        """
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
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
            _cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Normalize and tokenize text
        text = punc_norm(text)
        logger.info(f"Generating speech for: '{text[:50]}...' ({len(text)} chars)")

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
            speaker_emb=mx.array(self.conds.t3.speaker_emb.cpu().numpy()) if self.conds.t3.speaker_emb is not None else None,
            cond_prompt_speech_tokens=mx.array(self.conds.t3.cond_prompt_speech_tokens.cpu().numpy()) if self.conds.t3.cond_prompt_speech_tokens is not None else None,
            emotion_adv=float(self.conds.t3.emotion_adv[0, 0, 0].item()),
        )

        # Generate speech tokens with T3 MLX
        logger.info("Generating speech tokens with T3 MLX...")
        with torch.inference_mode():
            speech_tokens_mx = self.t3.inference(
                t3_cond=t3_cond_mx,
                text_tokens=text_tokens_mx,
                max_new_tokens=self.t3.hp.max_speech_tokens,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )

        # Extract conditional batch (first one)
        speech_tokens = speech_tokens_mx[0] if len(speech_tokens_mx.shape) > 1 else speech_tokens_mx

        # Convert back to PyTorch for S3Gen
        speech_tokens_np = np.array(speech_tokens).astype(np.int64)
        speech_tokens_pt = torch.from_numpy(speech_tokens_np)

        # Drop invalid tokens (SOS/EOS) - needs PyTorch tensor
        speech_tokens_pt = drop_invalid_tokens(speech_tokens_pt)

        # Filter out tokens >= 6561 (special tokens)
        speech_tokens_pt = speech_tokens_pt[speech_tokens_pt < 6561]
        speech_tokens_pt = speech_tokens_pt.to(self.device)

        # Generate waveform with S3Gen (PyTorch/MPS)
        logger.info(f"Generating waveform with S3Gen (PyTorch on {self.device})...")
        with torch.inference_mode():
            wav, sources = self.s3gen.inference(
                speech_tokens=speech_tokens_pt,
                ref_dict=self.conds.gen,
            )
            if sources is not None:
                del sources
            wav = wav.squeeze(0).detach().cpu().numpy()

        # Force memory cleanup after generation
        clear_device_memory()

        return torch.from_numpy(wav).unsqueeze(0)

    def _split_text_intelligently(self, text: str, language_id: str, target_words_per_chunk: int = 50) -> List[str]:
        """
        Split text at sentence/phrase boundaries for chunked generation.

        Args:
            text: Input text to split
            language_id: Language code for language-specific splitting
            target_words_per_chunk: Target number of words per chunk

        Returns:
            List of text chunks
        """
        # Define sentence endings based on language
        if language_id in ['ja', 'zh']:
            # Japanese and Chinese sentence endings
            sentence_pattern = r'[。！？\.!?]+'
        else:
            # Most other languages - use lookbehind to split AFTER punctuation
            sentence_pattern = r'(?<=[.!?])\s+'

        # Split into sentences
        sentences = [s.strip() for s in re.split(sentence_pattern, text) if s.strip()]

        # Group sentences into chunks of approximately target_words_per_chunk
        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            if not sentence.strip():
                continue
            word_count = len(sentence.split())

            if current_word_count + word_count > target_words_per_chunk and current_chunk:
                # Start new chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_word_count = word_count
            else:
                current_chunk.append(sentence)
                current_word_count += word_count

        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks if chunks else [text]

    def _crossfade_chunks(self, chunks: List[torch.Tensor], overlap_duration: float = 1.0) -> np.ndarray:
        """
        Concatenate audio chunks with crossfading.
        
        Args:
            chunks: List of audio torch tensors
            overlap_duration: Duration of crossfade in seconds
        
        Returns:
            Single concatenated audio as numpy array
        """
        if len(chunks) == 0:
            return np.array([], dtype=np.float32)
        if len(chunks) == 1:
            chunk = chunks[0]
            if isinstance(chunk, torch.Tensor):
                chunk_np = chunk.detach().cpu().numpy()
            else:
                chunk_np = chunk
            if chunk_np.ndim == 2:
                chunk_np = chunk_np.squeeze(0)
            return chunk_np

        overlap_samples = int(overlap_duration * self.sr)

        # Convert all chunks to numpy
        np_chunks = []
        for chunk in chunks:
            if isinstance(chunk, torch.Tensor):
                chunk_np = chunk.detach().cpu().numpy()
            else:
                chunk_np = chunk
            if chunk_np.ndim == 2:
                chunk_np = chunk_np.squeeze(0)
            np_chunks.append(chunk_np)

        # Build list of segments, concatenate once at end
        segments_to_concat = []
        
        # Process first chunk
        first_chunk = np_chunks[0]
        if len(first_chunk) > overlap_samples and len(np_chunks) > 1:
            segments_to_concat.append(first_chunk[:-overlap_samples])
            prev_overlap_region = first_chunk[-overlap_samples:]
        else:
            prev_overlap_region = None
            segments_to_concat.append(first_chunk)

        # Process middle and last chunks
        for i in range(1, len(np_chunks)):
            current_chunk = np_chunks[i]
            is_last = (i == len(np_chunks) - 1)
            
            if prev_overlap_region is not None and len(current_chunk) >= overlap_samples:
                # Create crossfade
                fade_out = np.linspace(1.0, 0.0, overlap_samples, dtype=np.float32)
                fade_in = np.linspace(0.0, 1.0, overlap_samples, dtype=np.float32)
                
                overlap_mixed = prev_overlap_region * fade_out + current_chunk[:overlap_samples] * fade_in
                segments_to_concat.append(overlap_mixed)
                
                if is_last:
                    segments_to_concat.append(current_chunk[overlap_samples:])
                else:
                    if len(current_chunk) > 2 * overlap_samples:
                        segments_to_concat.append(current_chunk[overlap_samples:-overlap_samples])
                    prev_overlap_region = current_chunk[-overlap_samples:]
            else:
                segments_to_concat.append(current_chunk)
                prev_overlap_region = None if is_last else (
                    current_chunk[-overlap_samples:] if len(current_chunk) >= overlap_samples else None
                )

        # Single concatenation at the end
        result = np.concatenate(segments_to_concat)
        return result

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
        chunk_size_words: int = 50,
        overlap_duration: float = 1.0,
        progress_callback=None,
    ) -> torch.Tensor:
        """
        Generate long audio by chunking text with crossfading between chunks.

        Args:
            text: Input text to synthesize (any length)
            language_id: Language code (e.g., 'en', 'es', 'ja', 'zh')
            audio_prompt_path: Path to reference audio file for voice cloning
            exaggeration: Voice exaggeration/expressiveness level (0.0-1.0)
            cfg_weight: Classifier-free guidance weight (0.0-1.0)
            temperature: Sampling temperature for token generation (0.0-2.0)
            repetition_penalty: Penalty for repeating tokens (1.0-3.0)
            min_p: Minimum probability threshold for sampling
            top_p: Nucleus sampling threshold
            chunk_size_words: Target number of words per chunk
            overlap_duration: Duration in seconds of crossfade between chunks
            progress_callback: Optional callback function for progress monitoring

        Returns:
            torch.Tensor: Generated audio waveform with shape (1, num_samples)
        """
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
                progress_callback(stage="preparing_conditionals", audio_path=audio_prompt_path)
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Split text into chunks
        text_chunks = self._split_text_intelligently(text, language_id, target_words_per_chunk=chunk_size_words)

        logger.info(f"Split text into {len(text_chunks)} chunks")
        for i, chunk in enumerate(text_chunks):
            logger.info(f"  Chunk {i+1}: {len(chunk.split())} words - '{chunk[:50]}...'")

        if progress_callback:
            progress_callback(
                stage="text_split",
                total_chunks=len(text_chunks),
                chunk_previews=[(i+1, len(chunk.split()), chunk[:50]) for i, chunk in enumerate(text_chunks)]
            )

        all_audio_chunks = []
        current_conditioning = audio_prompt_path

        # Generate each chunk
        for i, text_chunk in enumerate(text_chunks):
            logger.info(f"Generating chunk {i+1}/{len(text_chunks)}...")

            if progress_callback:
                progress_callback(
                    stage="chunk_start",
                    chunk_index=i,
                    chunk_number=i+1,
                    total_chunks=len(text_chunks),
                    text_preview=text_chunk[:50],
                    word_count=len(text_chunk.split())
                )

            # Generate audio for this chunk
            audio = self.generate(
                text_chunk,
                language_id,
                audio_prompt_path=current_conditioning,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )

            # Move to CPU immediately to free MPS memory
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu()
            all_audio_chunks.append(audio)

            # Aggressive memory cleanup after each chunk
            clear_device_memory()

            if progress_callback:
                progress_callback(
                    stage="chunk_complete",
                    chunk_index=i,
                    chunk_number=i+1,
                    total_chunks=len(text_chunks),
                    audio_shape=audio.shape
                )

        # Crossfade and concatenate all chunks
        logger.info(f"Crossfading {len(all_audio_chunks)} chunks...")

        if progress_callback:
            progress_callback(
                stage="crossfading",
                total_chunks=len(all_audio_chunks),
                overlap_duration=overlap_duration
            )

        final_audio = self._crossfade_chunks(all_audio_chunks, overlap_duration)

        if progress_callback:
            progress_callback(
                stage="complete",
                total_chunks=len(all_audio_chunks),
                final_audio_shape=(1, len(final_audio))
            )

        return torch.from_numpy(final_audio).unsqueeze(0)
