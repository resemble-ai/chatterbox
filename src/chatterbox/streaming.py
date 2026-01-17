# Copyright (c) 2026 Wonderful AI
# MIT License
"""
Streaming TTS interface for Chatterbox.

This module provides real-time streaming audio generation by yielding audio chunks
as speech tokens are generated, rather than waiting for the full generation to complete.

Key features:
- Configurable chunk size (default: 5 tokens = 200ms audio)
- Uses existing S3Gen streaming infrastructure (finalize parameter, HiFiGAN cache)
- Encoder output caching to avoid redundant computation during streaming
- Tracks latency metrics (TTFA, chunk timing)
- Integrates with AlignmentStreamAnalyzer for quality control

Usage:
    from chatterbox.streaming import ChatterboxStreamer, AudioChunk
    
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    streamer = ChatterboxStreamer(model, chunk_tokens=5)
    
    for chunk in streamer.generate("مرحباً", language_id="ar"):
        play_audio(chunk.audio)
"""

import time
import logging
from dataclasses import dataclass
from typing import Generator, Optional, List, TYPE_CHECKING

import torch
import torch.nn.functional as F

from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR

if TYPE_CHECKING:
    from .mtl_tts import ChatterboxMultilingualTTS

logger = logging.getLogger(__name__)

# Constants
S3_TOKEN_RATE = 25  # tokens per second
MS_PER_TOKEN = 1000 / S3_TOKEN_RATE  # 40ms per token


@dataclass
class AudioChunk:
    """A chunk of generated audio with metadata."""
    
    # Audio samples as a 1D tensor
    audio: torch.Tensor
    
    # Sample rate of the audio (24000 Hz)
    sample_rate: int
    
    # Whether this is the final chunk
    is_final: bool
    
    # Number of speech tokens in this chunk
    num_tokens: int
    
    # Chunk index (0-based)
    chunk_index: int
    
    # Time from start of generation to this chunk being ready (ms)
    latency_ms: float
    
    # Duration of audio in this chunk (ms)
    audio_duration_ms: float
    
    # Cumulative tokens generated so far
    total_tokens: int
    
    @property
    def audio_numpy(self):
        """Return audio as numpy array."""
        return self.audio.cpu().numpy()


@dataclass 
class StreamingMetrics:
    """Metrics collected during streaming generation."""
    
    # Time to first audio chunk (ms)
    ttfa_ms: float = 0.0
    
    # Total generation time (ms)
    total_time_ms: float = 0.0
    
    # Total audio duration generated (ms)
    total_audio_ms: float = 0.0
    
    # Number of chunks generated
    num_chunks: int = 0
    
    # Total speech tokens generated
    total_tokens: int = 0
    
    # Average chunk latency (ms)
    avg_chunk_latency_ms: float = 0.0
    
    # Real-time factor (generation time / audio duration)
    real_time_factor: float = 0.0
    
    def __str__(self):
        return (
            f"StreamingMetrics(\n"
            f"  TTFA: {self.ttfa_ms:.1f}ms\n"
            f"  Total time: {self.total_time_ms:.1f}ms\n"
            f"  Audio duration: {self.total_audio_ms:.1f}ms\n"
            f"  Chunks: {self.num_chunks}\n"
            f"  Tokens: {self.total_tokens}\n"
            f"  Avg chunk latency: {self.avg_chunk_latency_ms:.1f}ms\n"
            f"  Real-time factor: {self.real_time_factor:.2f}x\n"
            f")"
        )


class ChatterboxStreamer:
    """
    Streaming TTS interface that yields audio chunks during generation.
    
    This class wraps ChatterboxMultilingualTTS to provide streaming output
    by buffering speech tokens and converting them to audio in chunks.
    
    Args:
        model: ChatterboxMultilingualTTS instance
        chunk_tokens: Number of tokens per chunk (default: 5, ~200ms audio)
        overlap_tokens: Tokens to overlap between chunks for continuity (default: 1)
        min_first_chunk: Minimum tokens before first chunk (default: same as chunk_tokens)
    """
    
    # Default chunk size: 5 tokens = 200ms of audio
    DEFAULT_CHUNK_TOKENS = 5
    
    # Overlap between chunks for audio continuity
    DEFAULT_OVERLAP_TOKENS = 1
    
    def __init__(
        self,
        model: 'ChatterboxMultilingualTTS',
        chunk_tokens: int = None,
        overlap_tokens: int = None,
        min_first_chunk: int = None,
    ):
        self.model = model
        self.t3 = model.t3
        self.s3gen = model.s3gen
        self.tokenizer = model.tokenizer
        self.device = model.device
        
        self.chunk_tokens = chunk_tokens or self.DEFAULT_CHUNK_TOKENS
        self.overlap_tokens = overlap_tokens or self.DEFAULT_OVERLAP_TOKENS
        self.min_first_chunk = min_first_chunk or self.chunk_tokens
        
        # Validate parameters
        if self.chunk_tokens < 2:
            raise ValueError("chunk_tokens must be at least 2")
        if self.overlap_tokens >= self.chunk_tokens:
            raise ValueError("overlap_tokens must be less than chunk_tokens")
        
        # Cache for speaker embeddings
        self._speaker_cache = {}
        
        # Metrics from last generation
        self.last_metrics: Optional[StreamingMetrics] = None
        
        logger.info(
            f"ChatterboxStreamer initialized: "
            f"chunk_tokens={self.chunk_tokens}, "
            f"overlap_tokens={self.overlap_tokens}"
        )
    
    def generate(
        self,
        text: str,
        language_id: str = "ar",
        audio_prompt_path: str = None,
        speaker_id: str = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 2.0,
        min_p: float = 0.05,
        top_p: float = 1.0,
        max_tokens: int = 1000,
    ) -> Generator[AudioChunk, None, None]:
        """
        Generate audio in streaming chunks.
        
        Yields AudioChunk objects as soon as enough tokens are available.
        
        Args:
            text: Text to synthesize
            language_id: Language code (e.g., "ar", "en")
            audio_prompt_path: Path to reference audio for voice cloning
            speaker_id: Optional speaker ID for caching embeddings
            exaggeration: Emotion exaggeration factor (0.0-1.0)
            cfg_weight: Classifier-free guidance weight
            temperature: Sampling temperature
            repetition_penalty: Penalty for token repetition
            min_p: Minimum probability threshold
            top_p: Top-p (nucleus) sampling threshold
            max_tokens: Maximum speech tokens to generate
            
        Yields:
            AudioChunk objects containing audio samples and metadata
        """
        start_time = time.perf_counter()
        metrics = StreamingMetrics()
        chunk_latencies = []
        
        # Prepare conditioning
        if audio_prompt_path:
            # Check speaker cache first
            if speaker_id and speaker_id in self._speaker_cache:
                self.model.conds = self._speaker_cache[speaker_id]
                logger.debug(f"Using cached speaker embedding for {speaker_id}")
            else:
                self.model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
                if speaker_id:
                    self._speaker_cache[speaker_id] = self.model.conds
                    logger.debug(f"Cached speaker embedding for {speaker_id}")
        else:
            assert self.model.conds is not None, (
                "Please call prepare_conditionals() first or specify audio_prompt_path"
            )
        
        # Update exaggeration if needed
        if float(exaggeration) != float(self.model.conds.t3.emotion_adv[0, 0, 0].item()):
            from .models.t3.modules.cond_enc import T3Cond
            _cond = self.model.conds.t3
            self.model.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)
        
        # Normalize and tokenize text
        from .mtl_tts import punc_norm
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(
            text, language_id=language_id.lower() if language_id else None
        ).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # For CFG
        
        # Add SOT/EOT tokens
        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)
        
        # Initialize streaming state
        all_tokens: List[int] = []  # Cumulative tokens for full sequence
        last_processed_count = 0  # Track how many tokens we've converted to audio
        last_audio_samples = 0  # Track how many audio samples we've yielded
        vocoder_cache = torch.zeros(1, 1, 0).to(self.device)
        encoder_cache: Optional[dict] = None  # Encoder output cache for efficiency
        chunk_index = 0
        first_chunk_yielded = False
        
        # Minimum tokens needed to produce output (pre_lookahead_len + buffer)
        # S3Gen holds back 3 tokens when finalize=False
        min_tokens_for_output = 6  # Need at least this many before we can get any audio
        
        # Generate tokens using T3's streaming generator
        with torch.inference_mode():
            for token in self.t3.inference_streaming(
                t3_cond=self.model.conds.t3,
                text_tokens=text_tokens,
                cfg_weight=cfg_weight,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                max_new_tokens=max_tokens,
            ):
                all_tokens.append(token)
                
                # Calculate how many new tokens we have since last chunk
                new_tokens = len(all_tokens) - last_processed_count
                
                # Check if we have enough new tokens for a chunk
                min_new = max(self.min_first_chunk, min_tokens_for_output) if not first_chunk_yielded else self.chunk_tokens
                
                if new_tokens >= min_new and len(all_tokens) >= min_tokens_for_output:
                    chunk_start = time.perf_counter()
                    
                    # Process ALL tokens accumulated so far (S3Gen needs full sequence)
                    # Encoder caching ensures we only encode NEW tokens, not the full sequence
                    all_tokens_tensor = torch.tensor(
                        all_tokens, dtype=torch.long, device=self.device
                    ).unsqueeze(0)
                    
                    # Generate audio - use finalize=False for intermediate chunks
                    # (holds back 3 tokens for lookahead, avoids boundary artifacts)
                    # Encoder cache is passed to avoid re-encoding already processed tokens
                    full_audio, new_vocoder_cache, encoder_cache = self._process_chunk(
                        all_tokens_tensor,
                        vocoder_cache,
                        encoder_cache=encoder_cache,
                        finalize=False,
                    )
                    
                    chunk_time = (time.perf_counter() - chunk_start) * 1000
                    total_time = (time.perf_counter() - start_time) * 1000
                    chunk_latencies.append(chunk_time)
                    
                    # Extract only NEW audio (portion we haven't yielded yet)
                    full_audio = full_audio.squeeze()
                    total_samples = full_audio.shape[-1]
                    
                    if total_samples > last_audio_samples:
                        # Yield only the new portion
                        new_audio = full_audio[last_audio_samples:]
                        audio_duration = new_audio.shape[-1] / S3GEN_SR * 1000
                        
                        # Track TTFA
                        if not first_chunk_yielded:
                            metrics.ttfa_ms = total_time
                            first_chunk_yielded = True
                            logger.info(f"⚡ TTFA: {metrics.ttfa_ms:.1f}ms")
                        
                        yield AudioChunk(
                            audio=new_audio,
                            sample_rate=S3GEN_SR,
                            is_final=False,
                            num_tokens=new_tokens,
                            chunk_index=chunk_index,
                            latency_ms=total_time,
                            audio_duration_ms=audio_duration,
                            total_tokens=len(all_tokens),
                        )
                        
                        metrics.total_audio_ms += audio_duration
                        chunk_index += 1
                        last_audio_samples = total_samples
                    
                    last_processed_count = len(all_tokens)
                    vocoder_cache = new_vocoder_cache
        
        # Process final chunk with all remaining tokens
        if len(all_tokens) > 0:
            chunk_start = time.perf_counter()
            
            all_tokens_tensor = torch.tensor(
                all_tokens, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            
            # Final chunk with finalize=True (encoder cache will be cleared after this)
            full_audio, _, _ = self._process_chunk(
                all_tokens_tensor,
                vocoder_cache,
                encoder_cache=encoder_cache,
                finalize=True,
            )
            
            chunk_time = (time.perf_counter() - chunk_start) * 1000
            total_time = (time.perf_counter() - start_time) * 1000
            chunk_latencies.append(chunk_time)
            
            # Extract only NEW audio
            full_audio = full_audio.squeeze()
            total_samples = full_audio.shape[-1]
            
            if total_samples > last_audio_samples:
                new_audio = full_audio[last_audio_samples:]
                audio_duration = new_audio.shape[-1] / S3GEN_SR * 1000
                
                yield AudioChunk(
                    audio=new_audio,
                    sample_rate=S3GEN_SR,
                    is_final=True,
                    num_tokens=len(all_tokens) - last_processed_count,
                    chunk_index=chunk_index,
                    latency_ms=total_time,
                    audio_duration_ms=audio_duration,
                    total_tokens=len(all_tokens),
                )
                
                metrics.total_audio_ms += audio_duration
                chunk_index += 1
        
        # Finalize metrics
        metrics.total_time_ms = (time.perf_counter() - start_time) * 1000
        metrics.num_chunks = chunk_index
        metrics.total_tokens = len(all_tokens)
        metrics.avg_chunk_latency_ms = sum(chunk_latencies) / len(chunk_latencies) if chunk_latencies else 0
        metrics.real_time_factor = metrics.total_time_ms / metrics.total_audio_ms if metrics.total_audio_ms > 0 else 0
        
        self.last_metrics = metrics
        logger.info(f"Streaming complete: {metrics}")
    
    def _process_chunk(
        self,
        speech_tokens: torch.Tensor,
        vocoder_cache: torch.Tensor,
        encoder_cache: Optional[dict] = None,
        finalize: bool = False,
    ) -> tuple:
        """
        Convert speech tokens to audio using S3Gen with encoder caching.
        
        Args:
            speech_tokens: Tensor of speech token IDs (1, num_tokens)
            vocoder_cache: HiFiGAN cache from previous chunk
            encoder_cache: Optional encoder cache dict from previous chunk
            finalize: Whether this is the final chunk
            
        Returns:
            Tuple of (audio_tensor, updated_vocoder_cache, updated_encoder_cache)
        """
        # Clean tokens
        speech_tokens = drop_invalid_tokens(speech_tokens.squeeze())
        speech_tokens = speech_tokens.unsqueeze(0).to(self.device)
        
        if speech_tokens.numel() == 0:
            # Return silence if no valid tokens
            return torch.zeros(1, 960).to(self.device), vocoder_cache, encoder_cache
        
        # Flow inference (tokens -> mel) with encoder caching
        output_mels, new_encoder_cache = self.s3gen.flow_inference(
            speech_tokens,
            ref_dict=self.model.conds.gen,
            finalize=finalize,
            encoder_cache=encoder_cache,
        )
        
        # HiFiGAN inference (mel -> audio) with caching
        output_wav, new_vocoder_cache = self.s3gen.hift_inference(
            output_mels,
            cache_source=vocoder_cache,
        )
        
        return output_wav, new_vocoder_cache, new_encoder_cache
    
    def clear_speaker_cache(self):
        """Clear the speaker embedding cache."""
        self._speaker_cache.clear()
        logger.info("Speaker cache cleared")
    
    def get_cached_speakers(self) -> List[str]:
        """Return list of cached speaker IDs."""
        return list(self._speaker_cache.keys())
