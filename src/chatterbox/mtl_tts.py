# mtl_tts.py
# Copyright (c) 2025 Resemble AI
# MIT License

from dataclasses import dataclass
from pathlib import Path
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future
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
        ("â€¦", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("â€”", "-"),
        ("â€“", "-"),
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
    sentence_enders = {".", "!", "?", "-", ",", "ã€", "ï¼Œ", "ã€‚", "ï¼Ÿ", "ï¼"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


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

        # NOTE: Explicit .half() on s3gen.flow / s3gen.mel2wav was removed.
        # The CFM ODE solver and HiFiGAN create internal fp32 tensors (timestep embeddings,
        # noise vectors, etc.) that clash with fp16 weights. Instead, we use torch.autocast
        # at the inference call sites in _process_token_buffer(), which handles mixed dtypes
        # correctly by casting at op boundaries.

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
                    n_cfm_timesteps=4,  # OPTIMIZATION: 4 vs default 10 — 60% fewer ODE steps
                )
            wav = wav.squeeze(0).detach().float().cpu().numpy()
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
        #   [(12, 15), (13, 11), (9, 2)]  →  layers 12, 13, 9
        target_layer_indices = {9, 12, 13}
        
        # Check if already patched to avoid double-patching
        if getattr(self, '_attention_patched', False):
            return
        
        patched_count = 0
        for i, layer in enumerate(self.t3.tfmr.layers):
            if i in target_layer_indices:
                continue  # Leave target layers unchanged — they need attention weights
            
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

    def _ensure_t3_patched_model(self, len_cond: int, text_tokens_2d: torch.Tensor):
        """
        Mirrors the patched-model compilation logic in T3.inference(), but kept here so we
        can stream without modifying T3 itself.
        
        OPTIMIZED: Only builds patched model once, then caches it.
        IMPORTANT: Always creates a fresh AlignmentStreamAnalyzer per request because
        text_tokens_slice depends on the current text length (varies between requests).
        This matches the official T3.inference() behavior which rebuilds every time.
        """
        # Always create a fresh alignment analyzer — text_tokens_slice changes per request
        alignment_stream_analyzer = None
        if getattr(self.t3.hp, "is_multilingual", False):
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
            
            # OPTIMIZATION: Patch non-target layers to use SDPA fast path
            self._patch_selective_attention()
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
    ) -> Generator[torch.Tensor, None, None]:
        """
        Stream speech tokens from T3 using the same logic as T3.inference(), but yielding
        token chunks with adaptive sizing.
        
        Args:
            chunk_sizes: List of chunk sizes [first_chunk, second_chunk, ...]. 
                        If None, uses [25] for all chunks.
        
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

        # Ensure patched model is available
        self._ensure_t3_patched_model(len_cond=len_cond, text_tokens_2d=text_tokens)

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
        output = self.t3.patched_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=True,
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

            # Alignment analyzer (multilingual integrity checks)
            if getattr(self.t3.patched_model, "alignment_stream_analyzer", None) is not None:
                last_token = generated_ids[0, -1].item() if generated_ids.numel() else None
                logits = self.t3.patched_model.alignment_stream_analyzer.step(
                    logits, next_token=last_token
                )

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

            output = self.t3.patched_model(
                inputs_embeds=next_token_embed,
                past_key_values=past,
                output_attentions=True,
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
        Decode a token buffer into an audio chunk.
        
        OPTIMIZED:
        - Split flow/hifigan: CFM on all tokens, HiFiGAN only on new mels (~50% HiFiGAN savings)
        - Reduced CFM timesteps: 4 instead of default 10 (60% fewer ODE steps)
        - fp16 flow + hifigan: halved memory bandwidth on all decode ops
        
        Pipeline:
          1. CFM flow on [context_tokens + new_tokens] -> full mel spectrogram
          2. Crop mels to only the new portion (skip context mels)
          3. HiFiGAN on new mels only -> audio chunk
        """
        decode_start = time.time()
        
        new_tokens = torch.cat(token_buffer, dim=-1)  # 1D

        if all_tokens_so_far is not None and all_tokens_so_far.numel() > 0:
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

        # === SPLIT VOCODER PIPELINE ===
        # Step 1: Run CFM flow on ALL tokens (context + new) for mel coherence
        # OPTIMIZATION: Use reduced CFM timesteps (4 vs default 10) — 60% fewer ODE steps
        # OPTIMIZATION: autocast to fp16 — avoids explicit .half() which breaks internal fp32 intermediates
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
        
        # Step 2: Crop mels to only the NEW portion (skip context mels)
        if context_length > 0:
            total_tokens = int(clean_tokens.numel())
            total_mel_frames = output_mels.shape[-1]
            mel_per_token = total_mel_frames / max(total_tokens, 1)
            skip_mel_frames = int(context_length * mel_per_token)
            new_mels = output_mels[:, :, skip_mel_frames:]
        else:
            new_mels = output_mels

        if new_mels.shape[-1] == 0:
            return None, 0.0, False

        # Step 3: Run HiFiGAN ONLY on new mels (saves ~50% of HiFiGAN per chunk)
        with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=autocast_enabled):
            wav, _ = self.s3gen.hift_inference(new_mels)
        audio_chunk = wav.squeeze(0).detach().float().cpu().numpy()

        if len(audio_chunk) == 0:
            return None, 0.0, False

        # Apply trim_fade to suppress reference spillover (matches s3gen.inference behavior)
        # Cache the numpy version to avoid repeated GPU->CPU conversion
        if not hasattr(self, '_trim_fade_np'):
            self._trim_fade_np = self.s3gen.trim_fade.cpu().numpy()
        fade_len = min(len(self._trim_fade_np), len(audio_chunk))
        if metrics.chunk_count == 0 and fade_len > 0:
            audio_chunk[:fade_len] *= self._trim_fade_np[:fade_len]

        # Fade-in to soften chunk boundaries
        fade_samples = int(fade_duration * self.sr)
        if fade_samples > 0 and metrics.chunk_count > 0:
            fade_samples = min(fade_samples, len(audio_chunk))
            fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=audio_chunk.dtype)
            audio_chunk[:fade_samples] *= fade_in

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
        chunk_size: int = 25,
        first_chunk_size: int = 5,
        context_window: int = 25,
        fade_duration: float = 0.02,
        print_metrics: bool = False,
        max_new_tokens: int = 1000,
        max_history_tokens: int = 500,
        skip_watermark: bool = False,
        n_cfm_timesteps: int = 4,
    ) -> Generator[Tuple[torch.Tensor, StreamingMetrics], None, None]:
        """
        Streaming TTS that yields audio chunks as they are produced.
        
        OPTIMIZATIONS:
        - Split vocoder: CFM on all tokens, HiFiGAN only on new mels (~50% HiFiGAN savings)
        - Reduced CFM timesteps: 4 vs default 10 (60% fewer ODE solver passes)
        - fp16 S3Gen flow + HiFiGAN: halved memory bandwidth on decode
        - Concurrent pipeline: T3 generates next tokens WHILE vocoder decodes current chunk
        - Deferred watermarking: skip_watermark=True to apply watermark in server instead
        - Selective attention: only layer 9 uses eager attention, rest use fast SDPA
        - T3 in float16, autocast for generation loop
        - Reduced context_window (25 vs 50)

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

        # === STREAMING GENERATION WITH CONCURRENT PIPELINE ===
        total_audio_length = 0.0
        all_tokens_processed = torch.empty((0,), dtype=torch.long, device=self.device)
        
        chunk_sizes = [first_chunk_size, chunk_size]

        first_token_start = time.time()
        first_token_measured = False

        # OPTIMIZATION: Use autocast for mixed precision
        _dev_type = self.device.split(':')[0] if self.device != "cpu" else "cpu"
        _amp_dtype = torch.float16 if _dev_type == "cuda" else torch.bfloat16

        # Concurrent pipeline: decode chunk N in background while T3 generates chunk N+1
        # Use a dedicated CUDA stream for vocoder to overlap GPU work
        vocoder_stream = None
        if _dev_type == "cuda":
            vocoder_stream = torch.cuda.Stream()

        decode_executor = ThreadPoolExecutor(max_workers=1)
        pending_decode: Optional[Future] = None
        pending_tokens_for_history: Optional[torch.Tensor] = None

        def _decode_chunk(token_chunk, all_prev_tokens):
            """Run vocoder decode, optionally on a separate CUDA stream."""
            if vocoder_stream is not None:
                with torch.cuda.stream(vocoder_stream):
                    result = self._process_token_buffer(
                        [token_chunk],
                        all_prev_tokens,
                        context_window,
                        start_time,
                        metrics,
                        print_metrics,
                        fade_duration=fade_duration,
                        skip_watermark=skip_watermark,
                        n_cfm_timesteps=n_cfm_timesteps,
                    )
                vocoder_stream.synchronize()
                return result
            else:
                return self._process_token_buffer(
                    [token_chunk],
                    all_prev_tokens,
                    context_window,
                    start_time,
                    metrics,
                    print_metrics,
                    fade_duration=fade_duration,
                    skip_watermark=skip_watermark,
                    n_cfm_timesteps=n_cfm_timesteps,
                )

        try:
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
                ):
                    if not first_token_measured:
                        metrics.first_token_time = time.time() - first_token_start
                        if print_metrics:
                            print(f"Time to first token chunk: {metrics.first_token_time:.3f}s")
                        first_token_measured = True
                    
                    # === CONCURRENT PIPELINE ===
                    # If there's a pending decode from the previous iteration, 
                    # wait for it and yield the result. This means T3 was generating
                    # this chunk's tokens WHILE the previous chunk was being decoded.
                    if pending_decode is not None:
                        audio_tensor, audio_duration, success = pending_decode.result()
                        if success:
                            total_audio_length += audio_duration
                            yield audio_tensor, metrics
                        
                        # Update token history from the previous chunk
                        if pending_tokens_for_history is not None:
                            if all_tokens_processed.numel() == 0:
                                all_tokens_processed = pending_tokens_for_history
                            else:
                                all_tokens_processed = torch.cat([all_tokens_processed, pending_tokens_for_history], dim=-1)
                                if all_tokens_processed.numel() > max_history_tokens:
                                    all_tokens_processed = all_tokens_processed[-max_history_tokens:]

                    # Submit current chunk for background decoding
                    # Clone the token history snapshot so the background thread has stable data
                    if all_tokens_processed.numel() > context_window:
                        history_snapshot = all_tokens_processed[-context_window:].clone()
                    else:
                        history_snapshot = all_tokens_processed.clone()
                    pending_decode = decode_executor.submit(_decode_chunk, token_chunk_1d, history_snapshot)
                    pending_tokens_for_history = token_chunk_1d

            # === Flush the last pending decode ===
            if pending_decode is not None:
                audio_tensor, audio_duration, success = pending_decode.result()
                if success:
                    total_audio_length += audio_duration
                    yield audio_tensor, metrics

        finally:
            decode_executor.shutdown(wait=False)

        metrics.total_generation_time = time.time() - start_time
        metrics.total_audio_duration = total_audio_length
        if total_audio_length > 0:
            metrics.rtf = metrics.total_generation_time / total_audio_length
            if print_metrics:
                print(f"Total generation time: {metrics.total_generation_time:.3f}s")
                print(f"Total audio duration: {metrics.total_audio_duration:.3f}s")
                print(f"RTF (Real-Time Factor): {metrics.rtf:.3f}")
                print(f"Total chunks yielded: {metrics.chunk_count}")