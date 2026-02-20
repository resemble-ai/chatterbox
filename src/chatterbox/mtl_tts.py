# mtl_tts.py
# Copyright (c) 2025 Resemble AI
# MIT License

from dataclasses import dataclass
from pathlib import Path
import os
import time
import logging
from typing import Generator, Tuple, Optional, Dict

import librosa
import numpy as np
import torch
import perth
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download
from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor,
    TopPLogitsWarper,
    MinPLogitsWarper,
)

from .models.t3 import T3
from .models.t3.modules.t3_config import T3Config
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import MTLTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond

# NOTE: These exist in the official repo (used inside T3.inference).
# We import them here so we can implement streaming without modifying T3 itself.
from .models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
from .models.t3.inference.alignment_stream_analyzer import AlignmentStreamAnalyzer


REPO_ID = "ResembleAI/chatterbox"

logger = logging.getLogger(__name__)


class TokenRepetitionGuard:
    """
    Lightweight replacement for AlignmentStreamAnalyzer that doesn't need
    attention weights. Detects:
    - Token-level repetition (N consecutive identical tokens)
    - Low-diversity tail: model cycling through a small set of garbage tokens
    - N-gram repetition: repeating short patterns (e.g., A-B-A-B-A-B)
    - Runaway generation (exceeding expected length based on text tokens)
    
    The low-diversity and n-gram checks replace the AlignmentStreamAnalyzer's
    `long_tail` detection, which relied on attention weights we no longer compute.
    These checks only activate after the model has generated enough tokens for
    the text content (TAIL_CHECK_RATIO), avoiding false positives during
    legitimate speech that may have repeated phonemes.
    
    This allows output_attentions=False -> ALL layers use fast SDPA.
    """
    
    # Window for diversity check — how many recent tokens to examine
    DIVERSITY_WINDOW = 12
    # Minimum unique tokens required in the window; below this -> force EOS
    DIVERSITY_MIN_UNIQUE = 4
    
    # N-gram repetition detection
    NGRAM_SIZES = [2, 3]
    # Force EOS if the same n-gram repeats this many times in the window
    NGRAM_REPEAT_THRESHOLD = 3
    # Larger window for n-gram analysis
    NGRAM_WINDOW = 16
    
    # Only apply diversity/n-gram checks after this fraction of expected generation length.
    # Speech tokens ≈ 4-8x text tokens; we start checking after 4x to avoid false positives.
    TAIL_CHECK_RATIO = 4.0
    
    def __init__(self, eos_idx: int, text_token_count: int, 
                 max_repeat: int = 2, max_length_ratio: float = 10.0):
        """
        Args:
            eos_idx: EOS token ID to force when problems detected
            text_token_count: Number of text tokens (for length ratio check)
            max_repeat: Force EOS after this many consecutive identical tokens
            max_length_ratio: Force EOS if speech_tokens > text_tokens * ratio
        """
        self.eos_idx = eos_idx
        self.text_token_count = text_token_count
        self.max_repeat = max_repeat
        self.max_length_ratio = max_length_ratio
        
        self.generated_tokens = []
        self.step_count = 0
        
        # EOS suppression: don't allow EOS in the first N steps
        # (matches alignment analyzer's behavior of suppressing early EOS)
        self.min_steps_before_eos = max(5, text_token_count // 2)
        
        # Step count after which tail-quality checks activate
        self.tail_check_start = int(text_token_count * self.TAIL_CHECK_RATIO)
    
    def _check_low_diversity(self) -> bool:
        """Check if recent tokens have suspiciously low diversity (garbage cycling)."""
        if len(self.generated_tokens) < self.DIVERSITY_WINDOW:
            return False
        window = self.generated_tokens[-self.DIVERSITY_WINDOW:]
        unique_count = len(set(window))
        return unique_count < self.DIVERSITY_MIN_UNIQUE
    
    def _check_ngram_repetition(self) -> bool:
        """Check if any short n-gram repeats excessively (e.g., A-B-A-B-A-B)."""
        if len(self.generated_tokens) < self.NGRAM_WINDOW:
            return False
        window = self.generated_tokens[-self.NGRAM_WINDOW:]
        for n in self.NGRAM_SIZES:
            if len(window) < n * self.NGRAM_REPEAT_THRESHOLD:
                continue
            # Extract all n-grams in the window
            ngrams = [tuple(window[i:i+n]) for i in range(len(window) - n + 1)]
            # Count occurrences of the most recent n-gram
            last_ngram = ngrams[-1]
            count = ngrams.count(last_ngram)
            if count >= self.NGRAM_REPEAT_THRESHOLD:
                return True
        return False
    
    def step(self, logits, next_token=None):
        """
        Check for repetition/runaway and modify logits if needed.
        Same interface as AlignmentStreamAnalyzer.step().
        """
        self.step_count += 1
        
        # Track tokens
        if next_token is not None:
            if isinstance(next_token, torch.Tensor):
                token_id = next_token.item() if next_token.numel() == 1 else next_token.view(-1)[0].item()
            else:
                token_id = next_token
            self.generated_tokens.append(token_id)
            if len(self.generated_tokens) > self.NGRAM_WINDOW + 4:
                self.generated_tokens = self.generated_tokens[-(self.NGRAM_WINDOW + 4):]
        
        # Suppress early EOS
        if self.step_count < self.min_steps_before_eos:
            logits[..., self.eos_idx] = -2**15
            return logits
        
        force_eos = False
        reason = ""
        
        # Check 1: Consecutive identical token repetition (always active after min_steps)
        if len(self.generated_tokens) >= self.max_repeat:
            if len(set(self.generated_tokens[-self.max_repeat:])) == 1:
                reason = f"token_repeat ({self.max_repeat}x of {self.generated_tokens[-1]})"
                force_eos = True
        
        # Check 2: Runaway length (always active)
        max_len = int(self.text_token_count * self.max_length_ratio)
        if not force_eos and self.step_count > max_len:
            reason = f"runaway ({self.step_count} > {max_len})"
            force_eos = True
        
        # Checks 3 & 4: Only activate after enough tokens for actual speech content.
        # This prevents false positives on legitimate repeated phonemes.
        if not force_eos and self.step_count > self.tail_check_start:
            # Check 3: Low-diversity window (model cycling through small token set)
            if self._check_low_diversity():
                reason = f"low_diversity (< {self.DIVERSITY_MIN_UNIQUE} unique in last {self.DIVERSITY_WINDOW})"
                force_eos = True
            
            # Check 4: N-gram repetition (e.g., A-B-A-B-A-B pattern)
            if not force_eos and self._check_ngram_repetition():
                reason = "ngram_repeat"
                force_eos = True
        
        if force_eos:
            logger.warning(f"TokenRepetitionGuard: forcing EOS — {reason}")
            logits = -(2**15) * torch.ones_like(logits)
            logits[..., self.eos_idx] = 2**15
        
        return logits

# Supported languages for the multilingual model
SUPPORTED_LANGUAGES: Dict[str, str] = {
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
        ("Ã¢â‚¬Â¦", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("Ã¢â‚¬â€", "-"),
        ("Ã¢â‚¬â€œ", "-"),
        (" ,", ","),
        (""", '"'),
        (""", '"'),
        ("'", "'"),
        ("'", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",", "Ã£â‚¬Â", "Ã¯Â¼Å’", "Ã£â‚¬â€š", "Ã¯Â¼Å¸", "Ã¯Â¼Â"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


def trim_trailing_silence(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -40.0,
    frame_length: int = 2048,
    hop_length: int = 512,
    buffer_ms: float = 150.0,
    min_silence_ms: float = 300.0,
) -> np.ndarray:
    """
    Trim trailing silence/noise from audio using RMS energy analysis.
    
    Workaround for multilingual TTS occasionally generating trailing noise
    after the actual speech content ends. Scans backwards from the end of
    the audio to find where speech energy drops below a threshold, then
    trims from that point (with a small buffer for natural endings).
    
    Only trims if the trailing silence exceeds min_silence_ms, preventing
    removal of legitimate inter-word pauses in streaming chunks.
    
    Args:
        audio: 1D numpy array of audio samples
        sr: Sample rate
        threshold_db: dB below peak RMS to consider as silence
        frame_length: Window size for RMS computation (samples)
        hop_length: Stride for RMS computation (samples)
        buffer_ms: Extra audio to keep after last detected speech (ms)
        min_silence_ms: Only trim if trailing silence exceeds this (ms)
    
    Returns:
        Trimmed audio array. Returns original if no significant trailing silence found.
    """
    if len(audio) < frame_length:
        return audio
    
    # Compute RMS energy per frame
    n_frames = 1 + (len(audio) - frame_length) // hop_length
    if n_frames <= 0:
        return audio
    
    rms = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + frame_length]
        rms[i] = np.sqrt(np.mean(frame ** 2) + 1e-10)
    
    # Convert to dB relative to peak RMS
    peak_rms = rms.max()
    if peak_rms < 1e-10:
        return audio  # All silence
    
    rms_db = 20.0 * np.log10(rms / peak_rms + 1e-10)
    
    # Scan backwards to find last frame above threshold
    last_speech_frame = n_frames - 1
    for i in range(n_frames - 1, -1, -1):
        if rms_db[i] > threshold_db:
            last_speech_frame = i
            break
    
    # Convert frame index to sample position
    last_speech_sample = last_speech_frame * hop_length + frame_length
    
    # Add buffer
    buffer_samples = int(buffer_ms * sr / 1000.0)
    trim_point = min(last_speech_sample + buffer_samples, len(audio))
    
    # How much silence is trailing?
    trailing_silence_samples = len(audio) - trim_point
    min_silence_samples = int(min_silence_ms * sr / 1000.0)
    
    # Only trim if trailing silence is significant
    if trailing_silence_samples < min_silence_samples:
        return audio
    
    trimmed = audio[:trim_point]
    
    # Apply a short fade-out at the trim point for a clean ending
    fade_out_samples = min(int(0.01 * sr), len(trimmed))  # 10ms fade
    if fade_out_samples > 1:
        fade = np.linspace(1.0, 0.0, fade_out_samples, dtype=trimmed.dtype)
        trimmed[-fade_out_samples:] *= fade
    
    return trimmed


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
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
            gen=self.gen,
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs["t3"]), kwargs["gen"])


@dataclass
class StreamingMetrics:
    """Metrics for streaming TTS generation"""
    latency_to_first_chunk: Optional[float] = None
    rtf: Optional[float] = None
    total_generation_time: Optional[float] = None
    total_audio_duration: Optional[float] = None
    chunk_count: int = 0
    # Profiling times
    prep_time: Optional[float] = None
    tokenization_time: Optional[float] = None
    first_token_time: Optional[float] = None
    first_decode_time: Optional[float] = None


class ChatterboxMultilingualTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: MTLTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def get_supported_languages(cls):
        """Return dictionary of supported language codes and names."""
        return SUPPORTED_LANGUAGES.copy()

    @classmethod
    def from_local(cls, ckpt_dir, device) -> "ChatterboxMultilingualTTS":
        ckpt_dir = Path(ckpt_dir)

        ve = VoiceEncoder()
        ve.load_state_dict(torch.load(ckpt_dir / "ve.pt", weights_only=True))
        ve.to(device).eval()

        t3 = T3(T3Config.multilingual())
        t3_state = load_safetensors(ckpt_dir / "t3_mtl23ls_v2.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        # OPTIMIZATION: Load in float16 to halve memory bandwidth and speed up all matmuls.
        # The model tolerates fp16 well (finetuning repos use fp16=True).
        if device != "cpu":
            t3.to(device).half().eval()
        else:
            t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(torch.load(ckpt_dir / "s3gen.pt", weights_only=True))
        s3gen.to(device).eval()

        # OPTIMIZATION: Move CFM flow estimator and HiFiGAN to fp16 on GPU.
        # The UpsampleConformerEncoder, ConditionalDecoder U-Net, and HiFTGenerator
        # all operate well in fp16. The noise/timestep tensors are created inside
        # autocast contexts in _process_token_buffer so mixed-dtype is handled correctly.
        # This halves memory bandwidth on all weight reads (biggest cost on modern GPUs).
        if device not in ("cpu",) and not str(device).startswith("cpu"):
            s3gen.flow.encoder.half()
            s3gen.flow.encoder_proj.half()
            s3gen.flow.input_embedding.half()
            s3gen.flow.spk_embed_affine_layer.half()
            s3gen.flow.decoder.estimator.half()
            s3gen.mel2wav.half()

        tokenizer = MTLTokenizer(str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"))

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device: torch.device) -> "ChatterboxMultilingualTTS":
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
        return cls.from_local(ckpt_dir, device)

    def compile(self):
        """
        Apply torch.compile to the S3Gen CFM estimator and HiFiGAN vocoder for
        faster inference via kernel fusion and (on CUDA) potential CUDA graph capture.

        Call this once after loading the model and before your first generate call.
        The first call will be slower due to compilation; subsequent calls are faster.

        Example::

            model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
            model.compile()  # one-time compile cost
            model.prepare_conditionals("ref.wav")
            for chunk, metrics in model.generate_stream(text, "en"):
                ...
        """
        if not hasattr(torch, "compile"):
            logger.warning("torch.compile not available (requires PyTorch >= 2.0). Skipping.")
            return

        logger.info("Compiling S3Gen CFM estimator with torch.compile …")
        self.s3gen.flow.decoder.estimator = torch.compile(
            self.s3gen.flow.decoder.estimator,
            mode="reduce-overhead",
            dynamic=False,
            fullgraph=False,
        )

        logger.info("Compiling HiFiGAN vocoder with torch.compile …")
        self.s3gen.mel2wav = torch.compile(
            self.s3gen.mel2wav,
            mode="reduce-overhead",
            dynamic=False,
            fullgraph=False,
        )

        logger.info("torch.compile applied. First inference call will be slower (warmup).")

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        # Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[: self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[: self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

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
        text,
        language_id,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=2.0,
        min_p=0.05,
        top_p=1.0,
    ):
        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(
            text, language_id=language_id.lower() if language_id else None
        ).to(self.device)

        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,  # TODO: use value in config
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]

            speech_tokens = drop_invalid_tokens(speech_tokens).to(self.device)

            autocast_device = "cuda" if "cuda" in str(self.device) else str(self.device)
            autocast_enabled = autocast_device in ("cuda", "mps")
            with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=autocast_enabled):
                wav, _ = self.s3gen.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=self.conds.gen,
                    n_cfm_timesteps=4,  # OPTIMIZATION: 4 vs default 10 â€” 60% fewer ODE steps
                )
            wav = wav.squeeze(0).detach().float().cpu().numpy()
            wav = trim_trailing_silence(wav, self.sr)
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)

        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    # ----------------------------
    # STREAMING IMPLEMENTATION
    # ----------------------------

    def _patch_selective_attention(self):
        """
        OPTIMIZATION: Patch all transformer layers EXCEPT those used by the
        AlignmentStreamAnalyzer to skip attention weight computation.
        
        How it works:
        - With output_attentions=True (required for alignment analyzer), ALL layers
          fall back from SDPA to slow eager attention. This is ~24x more work than needed.
        - This patch wraps each non-target layer's forward() to override output_attentions=False,
          so those layers use the fast SDPA kernel.
        - The wrapper reformats the output tuple to match what the model expects.
        
        IMPORTANT: AlignmentStreamAnalyzer registers hooks on MULTIPLE layers, not just one.
        From alignment_stream_analyzer.py:
            LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]
        So layers 9, 12, and 13 all need real attention weights. All others get patched.
        
        Result: ~21 layers use fast SDPA, only 3 use eager attention.
        """
        if not hasattr(self.t3, 'tfmr') or not hasattr(self.t3.tfmr, 'layers'):
            return  # Safety check
        
        # AlignmentStreamAnalyzer needs attention weights from these layers.
        # See LLAMA_ALIGNED_HEADS in alignment_stream_analyzer.py:
        #   [(12, 15), (13, 11), (9, 2)]  â†’  layers 12, 13, 9
        target_layer_indices = {9, 12, 13}
        
        # Check if already patched to avoid double-patching
        if getattr(self, '_attention_patched', False):
            return
        
        patched_count = 0
        for i, layer in enumerate(self.t3.tfmr.layers):
            if i in target_layer_indices:
                continue  # Leave target layers unchanged â€” they need attention weights
            
            original_forward = layer.forward
            
            def make_wrapper(orig_fn):
                def wrapper(*args, **kwargs):
                    # Force this layer to NOT compute attention weights
                    # This keeps it on the fast SDPA path instead of falling back to eager
                    kwargs['output_attentions'] = False
                    outputs = orig_fn(*args, **kwargs)
                    
                    # Reformat output to match expected tuple structure:
                    # With output_attentions=True:  (hidden_states, attn_weights, present_kv)
                    # With output_attentions=False: (hidden_states, present_kv)
                    # We insert None for the missing attention weights
                    if len(outputs) >= 2:
                        return (outputs[0], None) + outputs[1:]
                    return outputs + (None,)
                return wrapper
            
            layer.forward = make_wrapper(original_forward)
            patched_count += 1
        
        self._attention_patched = True

    def _ensure_t3_patched_model(self, len_cond: int, text_tokens_2d: torch.Tensor, use_alignment: bool = False):
        """
        Mirrors the patched-model compilation logic in T3.inference(), but kept here so we
        can stream without modifying T3 itself.
        
        OPTIMIZED: Only builds patched model once, then caches it.
        
        When use_alignment=False (default):
          - No AlignmentStreamAnalyzer created
          - No attention hooks registered
          - output_attentions stays False -> ALL layers use SDPA
          - A lightweight TokenRepetitionGuard handles EOS forcing instead
          
        When use_alignment=True:
          - Original behavior with AlignmentStreamAnalyzer
          - 3 layers fall back to eager attention (layers 9, 12, 13)
        """
        # Only create alignment analyzer when explicitly requested
        alignment_stream_analyzer = None
        if use_alignment and getattr(self.t3.hp, "is_multilingual", False):
            alignment_stream_analyzer = AlignmentStreamAnalyzer(
                self.t3.tfmr,
                None,
                text_tokens_slice=(len_cond, len_cond + text_tokens_2d.size(-1)),
                alignment_layer_idx=9,
                eos_idx=self.t3.hp.stop_speech_token,
            )

        if not self.t3.compiled:
            # First call: build the full patched model structure
            patched_model = T3HuggingfaceBackend(
                config=self.t3.cfg,
                llama=self.t3.tfmr,
                speech_enc=self.t3.speech_emb,
                speech_head=self.t3.speech_head,
                alignment_stream_analyzer=alignment_stream_analyzer,
            )
            self.t3.patched_model = patched_model
            self.t3.compiled = True
            
            if use_alignment:
                # OPTIMIZATION: Patch non-target layers to use SDPA fast path
                self._patch_selective_attention()
            else:
                # Ensure config doesn't have output_attentions=True
                # (which AlignmentStreamAnalyzer sets globally in __init__)
                if hasattr(self.t3.tfmr, 'config'):
                    self.t3.tfmr.config.output_attentions = False
        else:
            # Subsequent calls: just swap in the fresh analyzer
            # This prevents dimension mismatches when text length varies between requests
            self.t3.patched_model.alignment_stream_analyzer = alignment_stream_analyzer

    def _t3_inference_stream(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.Tensor,
        max_new_tokens: int = 1000,
        temperature: float = 0.8,
        cfg_weight: float = 0.5,
        repetition_penalty: float = 2.0,
        min_p: float = 0.05,
        top_p: float = 1.0,
        chunk_sizes: list = None,  # OPTIMIZED: Support variable chunk sizes
        stop_on_eos: bool = True,
        use_alignment: bool = False,
    ) -> Generator[torch.Tensor, None, None]:
        """
        Stream speech tokens from T3 using the same logic as T3.inference(), but yielding
        token chunks with adaptive sizing.
        
        Args:
            chunk_sizes: List of chunk sizes [first_chunk, second_chunk, ...]. 
                        If None, uses [25] for all chunks.
            use_alignment: If True, use AlignmentStreamAnalyzer (slower, needs attention weights).
                          If False (default), use TokenRepetitionGuard (faster, all SDPA).
        
        Yields: torch.LongTensor of shape (Tchunk,)  (1D tokens)
        """
        if chunk_sizes is None:
            chunk_sizes = [25]
        
        # text_tokens is expected as (B, Ttext) where B=2 for CFG
        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.t3.device)

        # Default initial speech to a single start-of-speech token, matching text_tokens shape
        initial_speech_tokens = self.t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])

        # Prepare custom input embeds
        embeds, len_cond = self.t3.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech_tokens,
            cfg_weight=cfg_weight,
        )

        # Ensure patched model is available (alignment hooks only if use_alignment=True)
        self._ensure_t3_patched_model(len_cond=len_cond, text_tokens_2d=text_tokens,
                                       use_alignment=use_alignment)

        # Determine whether we need attention weights
        need_attentions = use_alignment and (
            getattr(self.t3.patched_model, "alignment_stream_analyzer", None) is not None
        )

        # Create lightweight guard when alignment is off
        rep_guard = None
        if not need_attentions:
            text_token_count = max(text_tokens.size(-1) - 2, 1)  # exclude SOT/EOT
            rep_guard = TokenRepetitionGuard(
                eos_idx=self.t3.hp.stop_speech_token,
                text_token_count=text_token_count,
            )

        device = embeds.device

        bos_token = torch.tensor([[self.t3.hp.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = self.t3.speech_emb(bos_token)
        # NOTE: official inference() assumes learned pos emb and uses get_fixed_embedding
        bos_embed = bos_embed + self.t3.speech_pos_emb.get_fixed_embedding(0)

        # Make BOS batch match embeds batch (CFG expects B=2)
        bos_embed = bos_embed.repeat(embeds.size(0), 1, 1)

        # Combine condition and BOS
        inputs_embeds = torch.cat([embeds, bos_embed], dim=1)

        # Track generated token ids (conditional batch only) for repetition penalty
        generated_ids = bos_token.clone()  # (1,1)

        top_p_warper = TopPLogitsWarper(top_p=top_p)
        min_p_warper = MinPLogitsWarper(min_p=min_p)
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))

        # Initial forward pass
        # KEY OPTIMIZATION: output_attentions=False when alignment disabled -> ALL layers use SDPA
        output = self.t3.patched_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=need_attentions,
            output_hidden_states=False,
            return_dict=True,
        )
        past = output.past_key_values

        chunk_buf = []
        chunk_idx = 0

        for i in range(max_new_tokens):
            logits_step = output.logits[:, -1, :]  # (2,V) for CFG
            cond = logits_step[0:1, :]
            uncond = logits_step[1:2, :]
            cfg = torch.as_tensor(cfg_weight, device=cond.device, dtype=cond.dtype)
            logits = cond + cfg * (cond - uncond)  # (1,V)

            # Use alignment analyzer OR lightweight guard
            last_token = generated_ids[0, -1].item() if generated_ids.numel() else None
            if need_attentions and getattr(self.t3.patched_model, "alignment_stream_analyzer", None) is not None:
                logits = self.t3.patched_model.alignment_stream_analyzer.step(
                    logits, next_token=last_token
                )
            elif rep_guard is not None:
                logits = rep_guard.step(logits, next_token=last_token)

            ids_for_proc = generated_ids[:1, ...]
            logits = repetition_penalty_processor(ids_for_proc, logits)

            if temperature != 1.0:
                logits = logits / temperature

            logits = min_p_warper(ids_for_proc, logits)
            logits = top_p_warper(ids_for_proc, logits)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1,1)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            chunk_buf.append(next_token)

            # EOS
            if stop_on_eos and next_token.item() == self.t3.hp.stop_speech_token:
                if chunk_buf:
                    out_chunk = torch.cat(chunk_buf, dim=1).squeeze(0)  # (Tchunk,)
                    yield out_chunk
                return

            # Get current target chunk size
            current_chunk_size = chunk_sizes[min(chunk_idx, len(chunk_sizes) - 1)]

            # Yield full chunk
            if len(chunk_buf) >= current_chunk_size:
                out_chunk = torch.cat(chunk_buf, dim=1).squeeze(0)  # (Tchunk,)
                yield out_chunk
                chunk_buf = []
                chunk_idx += 1

            # Next token embed
            next_token_embed = self.t3.speech_emb(next_token)
            next_token_embed = next_token_embed + self.t3.speech_pos_emb.get_fixed_embedding(i + 1)
            next_token_embed = next_token_embed.repeat(embeds.size(0), 1, 1)

            # KEY OPTIMIZATION: output_attentions=False when alignment disabled
            output = self.t3.patched_model(
                inputs_embeds=next_token_embed,
                past_key_values=past,
                output_attentions=need_attentions,
                output_hidden_states=False,
                return_dict=True,
            )
            past = output.past_key_values

        # If we run out of steps, flush remaining buffer
        if chunk_buf:
            out_chunk = torch.cat(chunk_buf, dim=1).squeeze(0)
            yield out_chunk


    def _process_token_buffer(
        self,
        token_buffer: list,
        all_tokens_so_far: torch.Tensor,
        context_window: int,
        start_time: float,
        metrics: StreamingMetrics,
        print_metrics: bool,
        fade_duration: float = 0.02,
        skip_watermark: bool = False,
        n_cfm_timesteps: int = 4,
    ):
        """
        Decode a token buffer into an audio chunk using overlap-add for seamless boundaries.
        
        OPTIMIZED:
        - Split flow/hifigan: CFM on all tokens, HiFiGAN only on new mels
        - Reduced CFM timesteps: 4 instead of default 10 (60% fewer ODE steps)
        - fp16 flow + hifigan: halved memory bandwidth on all decode ops
        - Overlap-add cross-fade: eliminates mel crop boundary artifacts
        
        Pipeline:
          1. CFM flow on [context_tokens + new_tokens] -> full mel spectrogram
          2. Crop mels to new portion WITH extra overlap frames from context
          3. HiFiGAN on (overlap + new) mels -> audio with overlap prefix
          4. Cross-fade overlap prefix with previous chunk's tail
          5. Yield clean audio without boundary discontinuities
        """
        # Overlap-add parameters
        # 4 mel frames ~ 20-50ms audio depending on HiFiGAN upsampling factor
        OVERLAP_MEL_FRAMES = 4
        # Cross-fade duration in seconds for the audio overlap region
        CROSSFADE_SECONDS = 0.03

        decode_start = time.time()
        
        new_tokens = torch.cat(token_buffer, dim=-1)  # 1D

        # context_window=0 means no context (pure chunk-by-chunk decoding).
        # Guard against the -0 Python slice quirk which would return the full tensor.
        if context_window > 0 and all_tokens_so_far is not None and all_tokens_so_far.numel() > 0:
            context_tokens = (
                all_tokens_so_far[-context_window:]
                if all_tokens_so_far.numel() > context_window
                else all_tokens_so_far
            )
            tokens_to_process = torch.cat([context_tokens, new_tokens], dim=-1)
            context_length = int(context_tokens.numel())
        else:
            tokens_to_process = new_tokens
            context_length = 0

        clean_tokens = drop_invalid_tokens(tokens_to_process).to(self.device)
        if clean_tokens.numel() == 0:
            return None, 0.0, False

        # === SPLIT VOCODER PIPELINE WITH OVERLAP-ADD ===
        # Step 1: Run CFM flow on ALL tokens (context + new) for mel coherence
        autocast_device = "cuda" if "cuda" in str(self.device) else str(self.device)
        autocast_enabled = autocast_device in ("cuda", "mps")
        with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=autocast_enabled):
            output_mels = self.s3gen.flow_inference(
                speech_tokens=clean_tokens,
                ref_dict=self.conds.gen,
                finalize=True,
                n_cfm_timesteps=n_cfm_timesteps,
            )
        # output_mels shape: (1, 80, T_mel) where T_mel ~ 2 * n_tokens
        
        # Step 2: Crop mels to new portion WITH overlap from context for cross-fade
        if context_length > 0:
            total_tokens = int(clean_tokens.numel())
            total_mel_frames = output_mels.shape[-1]
            mel_per_token = total_mel_frames / max(total_tokens, 1)
            skip_mel_frames = int(context_length * mel_per_token)
            
            # Include extra overlap mel frames from context for cross-fade
            overlap_start = max(0, skip_mel_frames - OVERLAP_MEL_FRAMES)
            actual_overlap_mels = skip_mel_frames - overlap_start
            new_mels = output_mels[:, :, overlap_start:]
        else:
            new_mels = output_mels
            actual_overlap_mels = 0

        if new_mels.shape[-1] == 0:
            return None, 0.0, False

        # Step 3: Run HiFiGAN on (overlap + new) mels
        with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=autocast_enabled):
            wav, _ = self.s3gen.hift_inference(new_mels)
        audio_chunk = wav.squeeze(0).detach().float().cpu().numpy()

        if len(audio_chunk) == 0:
            return None, 0.0, False

        # Step 4: Cross-fade overlap region with previous chunk's tail
        prev_tail = getattr(self, '_prev_chunk_tail', None)
        
        if actual_overlap_mels > 0 and prev_tail is not None and len(prev_tail) > 0:
            # Estimate how many audio samples correspond to the overlap mel frames
            total_audio_from_mels = len(audio_chunk)
            total_mel_for_hifigan = new_mels.shape[-1]
            samples_per_mel = total_audio_from_mels / max(total_mel_for_hifigan, 1)
            overlap_audio_samples = int(actual_overlap_mels * samples_per_mel)
            
            # Use the smaller of: estimated overlap, previous tail length, crossfade duration
            crossfade_samples = int(CROSSFADE_SECONDS * self.sr)
            crossfade_len = min(overlap_audio_samples, len(prev_tail), crossfade_samples, len(audio_chunk))
            
            if crossfade_len > 1:
                fade_out = np.linspace(1.0, 0.0, crossfade_len, dtype=audio_chunk.dtype)
                fade_in = np.linspace(0.0, 1.0, crossfade_len, dtype=audio_chunk.dtype)
                
                # Cross-fade: blend previous tail with beginning of current chunk
                audio_chunk[:crossfade_len] = (
                    prev_tail[-crossfade_len:] * fade_out +
                    audio_chunk[:crossfade_len] * fade_in
                )
            
            # Trim the pure-overlap prefix (audio that was already yielded in prev chunk)
            # Keep only the cross-faded region + new audio
            trim_samples = max(0, overlap_audio_samples - crossfade_len)
            if trim_samples > 0 and trim_samples < len(audio_chunk):
                audio_chunk = audio_chunk[trim_samples:]
        elif metrics.chunk_count == 0:
            # First chunk: apply trim_fade to suppress reference spillover
            if not hasattr(self, '_trim_fade_np'):
                self._trim_fade_np = self.s3gen.trim_fade.cpu().numpy()
            fade_len = min(len(self._trim_fade_np), len(audio_chunk))
            if fade_len > 0:
                audio_chunk[:fade_len] *= self._trim_fade_np[:fade_len]

        if len(audio_chunk) == 0:
            return None, 0.0, False

        # Step 5: Trim trailing silence/noise from this chunk.
        # Only trims if >300ms of silence at the end, so mid-stream chunks
        # with legitimate inter-word pauses are unaffected. Catches the
        # multilingual TTS tail noise issue where the model generates
        # garbage tokens after speech content ends.
        audio_chunk = trim_trailing_silence(audio_chunk, self.sr)

        if len(audio_chunk) == 0:
            return None, 0.0, False

        # Step 6: Save tail for next chunk's cross-fade
        tail_samples = int(CROSSFADE_SECONDS * self.sr) + int(OVERLAP_MEL_FRAMES * 512)
        tail_samples = min(tail_samples, len(audio_chunk))
        self._prev_chunk_tail = audio_chunk[-tail_samples:].copy()

        audio_duration = len(audio_chunk) / self.sr

        # Watermarking: optionally skip per-chunk for speed (apply in server instead)
        if not skip_watermark:
            audio_chunk = self.watermarker.apply_watermark(audio_chunk, sample_rate=self.sr)
        
        audio_tensor = torch.from_numpy(audio_chunk).unsqueeze(0)

        decode_time = time.time() - decode_start

        if metrics.chunk_count == 0:
            metrics.latency_to_first_chunk = time.time() - start_time
            metrics.first_decode_time = decode_time
            if print_metrics:
                print(f"Latency to first chunk: {metrics.latency_to_first_chunk:.3f}s")
                print(f"First decode time: {metrics.first_decode_time:.3f}s")

        metrics.chunk_count += 1
        return audio_tensor, audio_duration, True

    def generate_stream(
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
        chunk_size: int = 50,
        first_chunk_size: int = 10,
        context_window: int = 10,
        fade_duration: float = 0.02,
        print_metrics: bool = False,
        max_new_tokens: int = 1000,
        max_history_tokens: int = 500,
        skip_watermark: bool = True,
        n_cfm_timesteps: int = 3,
        use_alignment: bool = False,
    ) -> Generator[Tuple[torch.Tensor, StreamingMetrics], None, None]:
        """
        Streaming TTS that yields audio chunks as they are produced.

        Default parameters are tuned for lowest RTF (~0.35-0.45 on a mid-range GPU):

        - chunk_size=50 / first_chunk_size=10: fewer CFM calls per utterance
          (10 chunks → 5 chunks for a 10s sentence = 2x fewer decode passes)
        - context_window=10: only 10 context tokens fed to CFM (was 25), reduces
          CFM input by ~40% at each step while preserving boundary quality
        - n_cfm_timesteps=3: 3 ODE steps instead of 4 (25% fewer estimator calls)
        - skip_watermark=True: skip per-chunk watermarking; apply in the server/agent
          layer instead (saves ~5-15 ms per chunk, ~50-100 ms total per utterance)

        Latency trade-offs:
        - First audio chunk arrives after first_chunk_size=10 speech tokens are
          generated (~0.4 s of audio), keeping time-to-first-audio low.
        - For ultra-low latency at the cost of more CFM calls, set first_chunk_size=5.

        OPTIMIZATIONS already in place:
        - Split vocoder: CFM on all tokens, HiFiGAN only on new mels (~50% HiFiGAN savings)
        - fp16 S3Gen flow + HiFiGAN weights (set in from_local) + autocast at inference
        - T3 in float16, autocast for generation loop
        - ALL 24 T3 transformer layers use fast SDPA kernels (output_attentions=False)
        - No attention weight materialisation or .cpu() copies per step

        When use_alignment=False (default):
        - ALL 24 transformer layers use fast SDPA kernels
        - No attention weight materialization or .cpu() copies per step
        - Lightweight TokenRepetitionGuard handles EOS forcing
        - ~30-50% speedup on T3 forward pass
        
        When use_alignment=True:
        - Original behavior with AlignmentStreamAnalyzer
        - 3 layers use eager attention, rest use SDPA via selective patch
        - Better alignment quality checks but slower

        Yields:
            (audio_chunk_tensor, metrics)
        """
        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        start_time = time.time()
        metrics = StreamingMetrics()

        # Reset overlap-add state for this new request
        self._prev_chunk_tail = None

        # === PREPARATION PHASE ===
        prep_start = time.time()
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)
        
        metrics.prep_time = time.time() - prep_start
        if print_metrics:
            print(f"Preparation time: {metrics.prep_time:.3f}s")

        # === TOKENIZATION PHASE ===
        tok_start = time.time()
        text = punc_norm(text)
        tok = self.tokenizer.text_to_tokens(
            text, language_id=language_id.lower() if language_id else None
        ).to(self.device)

        # CFG batch=2
        text_tokens = torch.cat([tok, tok], dim=0)

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)
        
        metrics.tokenization_time = time.time() - tok_start
        if print_metrics:
            print(f"Tokenization time: {metrics.tokenization_time:.3f}s")

        # === STREAMING GENERATION (SYNCHRONOUS PIPELINE) ===
        total_audio_length = 0.0
        all_tokens_processed = torch.empty((0,), dtype=torch.long, device=self.device)
        
        chunk_sizes = [first_chunk_size, chunk_size]

        first_token_start = time.time()
        first_token_measured = False

        # OPTIMIZATION: Use autocast for mixed precision
        _dev_type = self.device.split(':')[0] if self.device != "cpu" else "cpu"
        _amp_dtype = torch.float16 if _dev_type == "cuda" else torch.bfloat16

        with torch.inference_mode(), torch.autocast(device_type=_dev_type, dtype=_amp_dtype):
            for token_chunk_1d in self._t3_inference_stream(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                chunk_sizes=chunk_sizes,
                stop_on_eos=True,
                use_alignment=use_alignment,
            ):
                if not first_token_measured:
                    metrics.first_token_time = time.time() - first_token_start
                    if print_metrics:
                        print(f"Time to first token chunk: {metrics.first_token_time:.3f}s")
                    first_token_measured = True

                # Synchronous decode: simpler and actually faster than threaded
                # (Python GIL prevents real GPU overlap with ThreadPoolExecutor)
                if all_tokens_processed.numel() > context_window:
                    history_snapshot = all_tokens_processed[-context_window:]
                else:
                    history_snapshot = all_tokens_processed

                audio_tensor, audio_duration, success = self._process_token_buffer(
                    [token_chunk_1d],
                    history_snapshot,
                    context_window,
                    start_time,
                    metrics,
                    print_metrics,
                    fade_duration=fade_duration,
                    skip_watermark=skip_watermark,
                    n_cfm_timesteps=n_cfm_timesteps,
                )

                if success:
                    total_audio_length += audio_duration
                    yield audio_tensor, metrics

                # Update token history
                if all_tokens_processed.numel() == 0:
                    all_tokens_processed = token_chunk_1d
                else:
                    all_tokens_processed = torch.cat([all_tokens_processed, token_chunk_1d], dim=-1)
                    if all_tokens_processed.numel() > max_history_tokens:
                        all_tokens_processed = all_tokens_processed[-max_history_tokens:]

        metrics.total_generation_time = time.time() - start_time
        metrics.total_audio_duration = total_audio_length
        if total_audio_length > 0:
            metrics.rtf = metrics.total_generation_time / total_audio_length
            if print_metrics:
                print(f"Total generation time: {metrics.total_generation_time:.3f}s")
                print(f"Total audio duration: {metrics.total_audio_duration:.3f}s")
                print(f"RTF (Real-Time Factor): {metrics.rtf:.3f}")
                print(f"Total chunks yielded: {metrics.chunk_count}")