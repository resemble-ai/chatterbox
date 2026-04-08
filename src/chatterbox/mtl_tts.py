import logging
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Generator, Optional, Tuple

import librosa
import numpy as np
import torch
import perth
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download

from .models.t3 import T3
from .models.t3.modules.t3_config import T3Config
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import MTLTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


logger = logging.getLogger(__name__)

REPO_ID = "ResembleAI/chatterbox"


@dataclass
class StreamingMetrics:
    """Metrics for streaming TTS generation"""
    latency_to_first_chunk: Optional[float] = None
    rtf: Optional[float] = None
    total_generation_time: Optional[float] = None
    total_audio_duration: Optional[float] = None
    chunk_count: int = 0


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
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",","、","，","。","？","！"}
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

    def to(self, device=None, dtype=None):
        self.t3 = self.t3.to(device=device, dtype=dtype)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                if dtype is not None and v.is_floating_point():
                    self.gen[k] = v.to(device=device, dtype=dtype)
                else:
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
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


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
    def from_local(cls, ckpt_dir, device, dtype=None) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(
            torch.load(ckpt_dir / "ve.pt", map_location=map_location, weights_only=True)
        )
        ve.to(device=device, dtype=dtype).eval()

        t3 = T3(T3Config.multilingual())
        t3_state = load_safetensors(ckpt_dir / "t3_mtl23ls_v2.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device=device, dtype=dtype).eval()
        # NOTE: torch.compile on T3 disabled — incompatible with this transformers
        # version (output_capturing.py loses torch reference). See PENDING.md.
        # t3.compile_for_inference(mode="default")

        s3gen = S3Gen()
        s3gen.load_state_dict(
            torch.load(ckpt_dir / "s3gen.pt", map_location=map_location, weights_only=True)
        )
        s3gen.to(device=device, dtype=dtype).eval()
        # NOTE: torch.compile on S3Gen disabled — too many graph breaks (.item() in
        # mask.py, dynamic shapes) cause recompilation overhead. See PENDING.md.
        # s3gen.compile_for_inference()

        tokenizer = MTLTokenizer(
            str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device=device, dtype=dtype)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device: torch.device, dtype=None) -> 'ChatterboxMultilingualTTS':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        ckpt_dir = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main",
                allow_patterns=["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"],
                token=os.getenv("HF_TOKEN"),
            )
        )
        return cls.from_local(ckpt_dir, device, dtype=dtype)
    
    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
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
        text_tokens = self.tokenizer.text_to_tokens(text, language_id=language_id.lower() if language_id else None).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,  # TODO: use the value in config
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    def _inference_stream(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.Tensor,
        language_id: Optional[str] = None,
        max_new_tokens: int = 1000,
        temperature: float = 0.8,
        cfg_weight: float = 0.5,
        repetition_penalty: float = 2.0,
        min_p: float = 0.05,
        top_p: float = 1.0,
        chunk_size: int = 25,
    ) -> Generator[torch.Tensor, None, None]:
        """
        Token-level streaming generator for the multilingual model.
        Yields speech token chunks of `chunk_size` as the T3 loop produces them.
        S3 token rate is 25 tokens/sec, so chunk_size=25 ≈ 1 second per chunk.
        Uses AlignmentStreamAnalyzer for hallucination suppression.
        """
        from transformers.generation.logits_process import (
            TopPLogitsWarper,
            MinPLogitsWarper,
            RepetitionPenaltyLogitsProcessor,
        )
        from .models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
        from .models.t3.inference.alignment_stream_analyzer import AlignmentStreamAnalyzer

        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)
        initial_speech_tokens = self.t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])

        embeds, len_cond = self.t3.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech_tokens,
            cfg_weight=cfg_weight,
        )

        # Multilingual model always uses AlignmentStreamAnalyzer
        alignment_stream_analyzer = AlignmentStreamAnalyzer(
            self.t3.tfmr,
            None,
            text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
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

        device = embeds.device
        bos_token = torch.tensor([[self.t3.hp.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = self.t3.speech_emb(bos_token)
        bos_embed = bos_embed + self.t3.speech_pos_emb.get_fixed_embedding(0)
        bos_embed = torch.cat([bos_embed, bos_embed])  # batch=2 for CFG

        inputs_embeds = torch.cat([embeds, bos_embed], dim=1)
        chunk_buffer = []
        stop_token = self.t3.hp.stop_speech_token

        # Pre-allocate token buffer to avoid O(N²) torch.cat growth
        generated_ids = torch.zeros(1, max_new_tokens + 1, dtype=torch.long, device=device)
        generated_ids[0, 0] = self.t3.hp.start_speech_token
        gen_pos = 1  # next write position

        top_p_warper = TopPLogitsWarper(top_p=top_p)
        min_p_warper = MinPLogitsWarper(min_p=min_p)
        rep_pen = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))

        output = patched_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        past = output.past_key_values

        for i in range(max_new_tokens):
            logits = output.logits[:, -1, :]

            # CFG combine
            logits_cond = logits[0:1]
            logits_uncond = logits[1:2]
            logits = logits_cond + cfg_weight * (logits_cond - logits_uncond)

            # Alignment stream analyzer: suppresses hallucinations / forces EOS when needed
            last_token = generated_ids[0, gen_pos - 1].item() if gen_pos > 0 else None
            logits = alignment_stream_analyzer.step(logits, next_token=last_token)

            ids_for_proc = generated_ids[:1, :gen_pos]
            logits = rep_pen(ids_for_proc, logits)
            if temperature == 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                if temperature != 1.0:
                    logits = logits / temperature
                logits = min_p_warper(ids_for_proc, logits)
                logits = top_p_warper(ids_for_proc, logits)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            chunk_buffer.append(next_token)
            generated_ids[0, gen_pos] = next_token.view(-1)
            gen_pos += 1

            # Defer EOS check to chunk boundaries (one sync per chunk instead of per token)
            if len(chunk_buffer) >= chunk_size:
                chunk_tokens = torch.cat(chunk_buffer, dim=1)
                eos_mask = (chunk_tokens.view(-1) == stop_token)
                if eos_mask.any().item():
                    eos_idx = eos_mask.nonzero(as_tuple=False)[0].item()
                    if eos_idx > 0:
                        yield chunk_tokens[:, :eos_idx]
                    break
                yield chunk_tokens
                chunk_buffer = []

            next_token_embed = self.t3.speech_emb(next_token)
            next_token_embed = next_token_embed + self.t3.speech_pos_emb.get_fixed_embedding(i + 1)
            next_token_embed = torch.cat([next_token_embed, next_token_embed])  # CFG

            output = patched_model(
                inputs_embeds=next_token_embed,
                past_key_values=past,
                output_attentions=False,
                return_dict=True,
            )
            past = output.past_key_values

        # Flush remaining buffer (max_new_tokens reached or final partial chunk)
        if chunk_buffer:
            chunk_tokens = torch.cat(chunk_buffer, dim=1)
            eos_mask = (chunk_tokens.view(-1) == stop_token)
            if eos_mask.any().item():
                eos_idx = eos_mask.nonzero(as_tuple=False)[0].item()
                if eos_idx > 0:
                    yield chunk_tokens[:, :eos_idx]
            else:
                yield chunk_tokens

    def _process_token_chunk(
        self,
        token_chunk: torch.Tensor,
        all_tokens_so_far: list,
        context_window: int,
        start_time: float,
        metrics: StreamingMetrics,
        fade_duration: float = 0.02,
        cfm_steps: int = 10,
    ) -> Tuple[Optional[torch.Tensor], float]:
        """
        Decode a speech token chunk to audio via S3Gen with context overlap for smooth boundaries.
        """
        import time

        if all_tokens_so_far is not None:
            context = all_tokens_so_far[-context_window:] if all_tokens_so_far.numel() > context_window else all_tokens_so_far
            tokens_to_decode = torch.cat([context, token_chunk], dim=-1)
            context_length = context.numel()
        else:
            tokens_to_decode = token_chunk
            context_length = 0

        clean = drop_invalid_tokens(tokens_to_decode).to(self.device)
        clean = clean[clean < 6561]
        if clean.numel() == 0:
            return None, 0.0

        wav, _ = self.s3gen.inference(speech_tokens=clean, ref_dict=self.conds.gen, n_cfm_timesteps=cfm_steps)
        wav = wav.squeeze(0).detach().cpu().numpy()

        if context_length > 0:
            samples_per_token = len(wav) / len(clean)
            skip = int(context_length * samples_per_token)
            audio_chunk = wav[skip:]
        else:
            audio_chunk = wav

        if len(audio_chunk) == 0:
            return None, 0.0

        fade_samples = min(int(fade_duration * self.sr), len(audio_chunk))
        if fade_samples > 0:
            audio_chunk[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples, dtype=audio_chunk.dtype)

        audio_duration = len(audio_chunk) / self.sr

        if metrics.chunk_count == 0:
            metrics.latency_to_first_chunk = time.time() - start_time

        metrics.chunk_count += 1

        watermarked = self.watermarker.apply_watermark(audio_chunk, sample_rate=self.sr)
        return torch.from_numpy(watermarked).unsqueeze(0), audio_duration

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
        context_window: int = 50,
        fade_duration: float = 0.02,
        cfm_steps: int = 10,
    ) -> Generator[Tuple[torch.Tensor, StreamingMetrics], None, None]:
        """
        Streaming multilingual TTS: yields (audio_chunk, metrics) as tokens are generated.
        chunk_size=25 ≈ 1 second of audio per chunk (S3 token rate is 25 tokens/sec).
        cfm_steps: number of CFM flow-matching steps (default 10, lower = faster).
        """
        import time

        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(f"Unsupported language_id '{language_id}'. Supported: {supported}")

        start_time = time.time()
        metrics = StreamingMetrics()

        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please call prepare_conditionals() first or pass audio_prompt_path"

        if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
            _cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        text = punc_norm(text)
        lang = language_id.lower() if language_id else None
        text_tokens = self.tokenizer.text_to_tokens(text, language_id=lang).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        total_audio = 0.0
        all_tokens = None
        t3_time_total = 0.0
        s3_time_total = 0.0

        # Pipeline: run S3Gen vocoder on a background thread + separate CUDA stream
        # so it overlaps with T3 generating the next token chunk.
        from concurrent.futures import ThreadPoolExecutor
        is_cuda = (self.device.type == 'cuda') if isinstance(self.device, torch.device) else ('cuda' in str(self.device))
        s3_stream = torch.cuda.Stream(device=self.device) if is_cuda else None

        def _decode_chunk(token_chunk, all_tokens_snapshot):
            """Run S3Gen decode on a separate CUDA stream."""
            s3_start = time.time()
            if s3_stream is not None:
                with torch.cuda.stream(s3_stream):
                    result = self._process_token_chunk(
                        token_chunk, all_tokens_snapshot, context_window,
                        start_time, metrics, fade_duration, cfm_steps=cfm_steps,
                    )
                s3_stream.synchronize()
            else:
                result = self._process_token_chunk(
                    token_chunk, all_tokens_snapshot, context_window,
                    start_time, metrics, fade_duration, cfm_steps=cfm_steps,
                )
            return (*result, time.time() - s3_start)

        with torch.inference_mode(), ThreadPoolExecutor(max_workers=1) as pool:
            pending_future = None
            t3_start = time.time()

            for token_chunk in self._inference_stream(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                language_id=lang,
                max_new_tokens=1000,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                chunk_size=chunk_size,
            ):
                torch.cuda.synchronize()
                t3_time_total += time.time() - t3_start

                token_chunk = token_chunk[0]  # extract conditional batch

                # Collect previous S3Gen result (blocks until background decode finishes)
                if pending_future is not None:
                    audio, duration, s3_elapsed = pending_future.result()
                    s3_time_total += s3_elapsed
                    if audio is not None:
                        total_audio += duration
                        yield audio, metrics

                # Submit current chunk for S3Gen decode in background
                pending_future = pool.submit(_decode_chunk, token_chunk, all_tokens)

                all_tokens = token_chunk if all_tokens is None else torch.cat([all_tokens, token_chunk], dim=-1)
                t3_start = time.time()

            # Collect final S3Gen result
            if pending_future is not None:
                audio, duration, s3_elapsed = pending_future.result()
                s3_time_total += s3_elapsed
                if audio is not None:
                    total_audio += duration
                    yield audio, metrics

        print(f"[PERF] T3 token gen: {t3_time_total:.3f}s | S3Gen vocoder: {s3_time_total:.3f}s (overlapped) | wall: {time.time()-start_time:.3f}s", flush=True)

        metrics.total_generation_time = time.time() - start_time
        metrics.total_audio_duration = total_audio
        if total_audio > 0:
            metrics.rtf = metrics.total_generation_time / total_audio
