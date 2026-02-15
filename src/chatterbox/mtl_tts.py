# mtl_tts.py
# Copyright (c) 2025 Resemble AI
# MIT License

from dataclasses import dataclass
from pathlib import Path
import os
import time
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
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
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
    sentence_enders = {".", "!", "?", "-", ",", "、", "，", "。", "？", "！"}
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
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(torch.load(ckpt_dir / "s3gen.pt", weights_only=True))
        s3gen.to(device).eval()

        tokenizer = MTLTokenizer(str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"))

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device: torch.device, warmup: bool = True) -> "ChatterboxMultilingualTTS":
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
        instance = cls.from_local(ckpt_dir, device)
        
        if warmup:
            print("🔥 Warming up models (one-time setup)...")
            instance._warmup_models()
            print("✓ Models ready for fast inference")
        
        return instance

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

    def _warmup_models(self):
        """
        Warmup both T3 and S3Gen to trigger compilation/GPU kernel initialization.
        This significantly reduces latency on the first real generation.
        """
        if self.conds is None:
            raise RuntimeError("Cannot warmup without conditionals. Call prepare_conditionals first or use built-in voice.")
        
        warmup_start = time.time()
        
        # Longer warmup text to avoid alignment analyzer issues
        warmup_text = "Hello, this is a warmup test for the model."
        
        try:
            # Tokenize
            warmup_tokens = self.tokenizer.text_to_tokens(warmup_text, language_id="en").to(self.device)
            warmup_tokens = torch.cat([warmup_tokens, warmup_tokens], dim=0)  # CFG batch
            
            sot = self.t3.hp.start_text_token
            eot = self.t3.hp.stop_text_token
            warmup_tokens = F.pad(warmup_tokens, (1, 0), value=sot)
            warmup_tokens = F.pad(warmup_tokens, (0, 1), value=eot)
            
            with torch.inference_mode():
                # Warmup T3 (generate ~20 tokens to trigger compilation)
                warmup_speech_tokens = []
                for i, token_chunk in enumerate(self._t3_inference_stream(
                    t3_cond=self.conds.t3,
                    text_tokens=warmup_tokens,
                    max_new_tokens=25,  # Enough tokens for alignment analyzer
                    temperature=0.8,
                    cfg_weight=0.5,
                    repetition_penalty=2.0,
                    min_p=0.05,
                    top_p=1.0,
                    chunk_sizes=[20],  # Larger chunk to ensure alignment matrix is big enough
                    stop_on_eos=True,
                )):
                    warmup_speech_tokens.append(token_chunk)
                    # Get at least one good chunk
                    if i >= 0 and torch.cat(warmup_speech_tokens, dim=-1).numel() >= 15:
                        break
                
                # Warmup S3Gen
                if warmup_speech_tokens:
                    warmup_tokens_cat = torch.cat(warmup_speech_tokens, dim=-1)
                    clean_tokens = drop_invalid_tokens(warmup_tokens_cat).to(self.device)
                    if clean_tokens.numel() > 0:
                        _, _ = self.s3gen.inference(
                            speech_tokens=clean_tokens,
                            ref_dict=self.conds.gen,
                        )
            
            warmup_time = time.time() - warmup_start
            print(f"  Warmup completed in {warmup_time:.2f}s")
            
        except Exception as e:
            # Warmup failed but don't crash - just warn
            print(f"  ⚠️  Warmup failed (first generation may be slower): {e}")
            print(f"  Continuing anyway...")


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

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)

        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    # ----------------------------
    # STREAMING IMPLEMENTATION
    # ----------------------------

    def _ensure_t3_patched_model(self, len_cond: int, text_tokens_2d: torch.Tensor):
        """
        Mirrors the patched-model compilation logic in T3.inference(), but kept here so we
        can stream without modifying T3 itself.
        
        OPTIMIZED: Only builds once, then caches the compiled model.
        """
        if not self.t3.compiled:
            alignment_stream_analyzer = None
            if getattr(self.t3.hp, "is_multilingual", False):
                alignment_stream_analyzer = AlignmentStreamAnalyzer(
                    self.t3.tfmr,
                    None,
                    text_tokens_slice=(len_cond, len_cond + text_tokens_2d.size(-1)),
                    alignment_layer_idx=9,
                    eos_idx=self.t3.hp.stop_speech_token,
                )

            patched_model = T3HuggingfaceBackend(
                config=self.t3.cfg,
                llama=self.t3.tfmr,
                speech_enc=self.t3.speech_emb,
                speech_head=self.t3.speech_head,
                alignment_stream_analyzer=alignment_stream_analyzer,
            )
            self.t3.patched_model = patched_model
            self.t3.compiled = True

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
            output_hidden_states=True,
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
                output_hidden_states=True,
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
        is_first_chunk: bool = False,  # NEW: Skip context on first chunk
    ):
        """
        Decode a token buffer into an audio chunk by running S3Gen on
        (context + new_tokens) and cropping context audio.
        
        For first chunk, skip context to minimize S3Gen overhead.
        """
        decode_start = time.time()
        
        new_tokens = torch.cat(token_buffer, dim=-1)  # 1D

        # OPTIMIZATION: Skip context on first chunk to reduce S3Gen overhead
        if is_first_chunk or all_tokens_so_far is None or all_tokens_so_far.numel() == 0:
            tokens_to_process = new_tokens
            context_length = 0
        else:
            context_tokens = (
                all_tokens_so_far[-context_window:]
                if all_tokens_so_far.numel() > context_window
                else all_tokens_so_far
            )
            tokens_to_process = torch.cat([context_tokens, new_tokens], dim=-1)
            context_length = int(context_tokens.numel())

        clean_tokens = drop_invalid_tokens(tokens_to_process).to(self.device)
        if clean_tokens.numel() == 0:
            return None, 0.0, False

        wav, _ = self.s3gen.inference(
            speech_tokens=clean_tokens,
            ref_dict=self.conds.gen,
        )
        wav = wav.squeeze(0).detach().cpu().numpy()

        # Crop away context portion
        if context_length > 0:
            samples_per_token = len(wav) / max(int(clean_tokens.numel()), 1)
            skip_samples = int(context_length * samples_per_token)
            audio_chunk = wav[skip_samples:]
        else:
            audio_chunk = wav

        if len(audio_chunk) == 0:
            return None, 0.0, False

        # Fade-in to soften boundaries (skip on first chunk as there's no boundary)
        if not is_first_chunk:
            fade_samples = int(fade_duration * self.sr)
            if fade_samples > 0:
                fade_samples = min(fade_samples, len(audio_chunk))
                fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=audio_chunk.dtype)
                audio_chunk[:fade_samples] *= fade_in

        audio_duration = len(audio_chunk) / self.sr
        watermarked_chunk = self.watermarker.apply_watermark(audio_chunk, sample_rate=self.sr)
        audio_tensor = torch.from_numpy(watermarked_chunk).unsqueeze(0)

        decode_time = time.time() - decode_start

        if metrics.chunk_count == 0:
            metrics.latency_to_first_chunk = time.time() - start_time
            metrics.first_decode_time = decode_time
            if print_metrics:
                print(f"⏱️ Latency to first chunk: {metrics.latency_to_first_chunk:.3f}s")
                print(f"⏱️ First decode time: {metrics.first_decode_time:.3f}s")

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
        first_chunk_size: int = 5,  # OPTIMIZED: Smaller first chunk for lower latency
        context_window: int = 50,
        fade_duration: float = 0.02,
        print_metrics: bool = False,
        max_new_tokens: int = 1000,
        max_history_tokens: int = 500,  # OPTIMIZED: Limit token history growth
    ) -> Generator[Tuple[torch.Tensor, StreamingMetrics], None, None]:
        """
        Streaming version of multilingual generate that yields audio chunks as they are produced.
        
        OPTIMIZED with:
        - Cached model compilation (no rebuild on each call)
        - Smaller first chunk (default 5 tokens vs 25)
        - Progressive chunk sizing
        - Detailed profiling metrics
        - Memory-efficient token history

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
            print(f"⏱️ Preparation time: {metrics.prep_time:.3f}s")

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
            print(f"⏱️ Tokenization time: {metrics.tokenization_time:.3f}s")

        # === STREAMING GENERATION ===
        total_audio_length = 0.0
        all_tokens_processed = torch.empty((0,), dtype=torch.long, device=self.device)
        
        # Progressive chunk sizing: start small, then grow
        chunk_sizes = [first_chunk_size, chunk_size]

        first_token_start = time.time()
        first_token_measured = False
        chunk_count = 0  # Track chunk count locally

        with torch.inference_mode():
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
                        print(f"⏱️ Time to first token: {metrics.first_token_time:.3f}s")
                    first_token_measured = True
                
                # token_chunk_1d: (Tchunk,)
                is_first = (chunk_count == 0)  # Flag for first chunk optimization
                
                audio_tensor, audio_duration, success = self._process_token_buffer(
                    [token_chunk_1d],
                    all_tokens_processed,
                    context_window,
                    start_time,
                    metrics,
                    print_metrics,
                    fade_duration=fade_duration,
                    is_first_chunk=is_first,  # Pass first chunk flag
                )

                if success:
                    total_audio_length += audio_duration
                    chunk_count += 1
                    yield audio_tensor, metrics

                # Update all tokens processed with memory limit
                if all_tokens_processed.numel() == 0:
                    all_tokens_processed = token_chunk_1d
                else:
                    all_tokens_processed = torch.cat([all_tokens_processed, token_chunk_1d], dim=-1)
                    # Trim history to prevent unbounded growth
                    if all_tokens_processed.numel() > max_history_tokens:
                        all_tokens_processed = all_tokens_processed[-max_history_tokens:]

        metrics.total_generation_time = time.time() - start_time
        metrics.total_audio_duration = total_audio_length
        if total_audio_length > 0:
            metrics.rtf = metrics.total_generation_time / total_audio_length
            if print_metrics:
                print(f"⏱️ Total generation time: {metrics.total_generation_time:.3f}s")
                print(f"⏱️ Total audio duration: {metrics.total_audio_duration:.3f}s")
                print(f"⏱️ RTF (Real-Time Factor): {metrics.rtf:.3f}")
                print(f"⏱️ Total chunks yielded: {metrics.chunk_count}")