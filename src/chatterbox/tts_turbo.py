import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple

import librosa
import numpy as np
import torch
import perth
import pyloudnorm as ln

from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from .models.t3 import T3
from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond
from .models.t3.modules.t3_config import T3Config
from .models.s3gen.const import S3GEN_SIL
import logging
logger = logging.getLogger(__name__)

REPO_ID = "ResembleAI/chatterbox-turbo"


@dataclass
class StreamingMetrics:
    """Metrics for streaming TTS generation"""
    latency_to_first_chunk: Optional[float] = None
    rtf: Optional[float] = None
    total_generation_time: Optional[float] = None
    total_audio_duration: Optional[float] = None
    chunk_count: int = 0


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
        ("…", ", "),
        (":", ","),
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
    sentence_enders = {".", "!", "?", "-", ","}
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


class ChatterboxTurboTTS:
    ENC_COND_LEN = 15 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
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
    def from_local(cls, ckpt_dir, device, dtype=None) -> 'ChatterboxTurboTTS':
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(ckpt_dir / "ve.safetensors")
        )
        ve.to(device=device, dtype=dtype).eval()

        # Turbo specific hp
        hp = T3Config(text_tokens_dict_size=50276)
        hp.llama_config_name = "GPT2_medium"
        hp.speech_tokens_dict_size = 6563
        hp.input_pos_emb = None
        hp.speech_cond_prompt_len = 375
        hp.use_perceiver_resampler = False
        hp.emotion_adv = False

        t3 = T3(hp)
        t3_state = load_file(ckpt_dir / "t3_turbo_v1.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        del t3.tfmr.wte
        t3.to(device=device, dtype=dtype).eval()
        # Turbo uses GPT2 (no StaticCache needed) — mode="default" uses inductor kernel
        # fusion without CUDA graphs, which is compatible with the dynamic KV cache.
        if device not in ("cpu", "mps"):
            t3.compile_for_inference(mode="default")

        s3gen = S3Gen(meanflow=True)
        weights = load_file(ckpt_dir / "s3gen_meanflow.safetensors")
        s3gen.load_state_dict(
            weights, strict=True
        )
        s3gen.to(device=device, dtype=dtype).eval()

        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if len(tokenizer) != 50276:
            print(f"WARNING: Tokenizer len {len(tokenizer)} != 50276")

        conds = None
        builtin_voice = ckpt_dir / "conds.pt"
        if builtin_voice.exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device=device, dtype=dtype)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device, dtype=None) -> 'ChatterboxTurboTTS':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        local_path = snapshot_download(
            repo_id=REPO_ID,
            token=os.getenv("HF_TOKEN") or None,
            # Optional: Filter to download only what you need
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]
        )

        return cls.from_local(local_path, device, dtype=dtype)

    def norm_loudness(self, wav, sr, target_lufs=-27):
        try:
            meter = ln.Meter(sr)
            loudness = meter.integrated_loudness(wav)
            gain_db = target_lufs - loudness
            gain_linear = 10.0 ** (gain_db / 20.0)
            if math.isfinite(gain_linear) and gain_linear > 0.0:
                wav = wav * gain_linear
        except Exception as e:
            print(f"Warning: Error in norm_loudness, skipping: {e}")

        return wav

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5, norm_loudness=True):
        ## Load and norm reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        assert len(s3gen_ref_wav) / _sr > 5.0, "Audio prompt must be longer than 5 seconds!"

        if norm_loudness:
            s3gen_ref_wav = self.norm_loudness(s3gen_ref_wav, _sr)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
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
        repetition_penalty=1.2,
        min_p=0.00,
        top_p=0.95,
        audio_prompt_path=None,
        exaggeration=0.0,
        cfg_weight=0.0,
        temperature=0.8,
        top_k=1000,
        norm_loudness=True,
    ):
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration, norm_loudness=norm_loudness)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        if cfg_weight > 0.0 or exaggeration > 0.0 or min_p > 0.0:
            logger.warning("CFG, min_p and exaggeration are not supported by Turbo version and will be ignored.")

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_tokens = text_tokens.input_ids.to(self.device)

        speech_tokens = self.t3.inference_turbo(
            t3_cond=self.conds.t3,
            text_tokens=text_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Remove OOV tokens and add silence to end
        speech_tokens = speech_tokens[speech_tokens < 6561]
        speech_tokens = speech_tokens.to(self.device)
        silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]).long().to(self.device)
        speech_tokens = torch.cat([speech_tokens, silence])

        wav, _ = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=self.conds.gen,
            n_cfm_timesteps=2,
        )
        wav = wav.squeeze(0).detach().cpu().numpy()
        watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    def _process_token_chunk(
        self,
        token_chunk: torch.Tensor,
        all_tokens_so_far: list,
        context_window: int,
        start_time: float,
        metrics: StreamingMetrics,
        fade_duration: float = 0.02,
    ) -> Tuple[Optional[torch.Tensor], float]:
        """
        Decode a speech token chunk to audio via S3Gen (MeanFlow, n_cfm_timesteps=2).
        Uses context overlap for smooth chunk boundaries.
        """
        import time

        if all_tokens_so_far is not None:
            context = all_tokens_so_far[-context_window:] if all_tokens_so_far.numel() > context_window else all_tokens_so_far
            tokens_to_decode = torch.cat([context, token_chunk], dim=-1)
            context_length = context.numel()
        else:
            tokens_to_decode = token_chunk
            context_length = 0

        clean = tokens_to_decode[tokens_to_decode < 6561].to(self.device)
        if clean.numel() == 0:
            return None, 0.0

        wav, _ = self.s3gen.inference(
            speech_tokens=clean,
            ref_dict=self.conds.gen,
            n_cfm_timesteps=2,  # MeanFlow
        )
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
        audio_prompt_path: Optional[str] = None,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        norm_loudness: bool = True,
        chunk_size: int = 25,
        context_window: int = 50,
        fade_duration: float = 0.02,
    ) -> Generator[Tuple[torch.Tensor, StreamingMetrics], None, None]:
        """
        Streaming Turbo TTS: yields (audio_chunk, metrics) as tokens are generated.
        No CFG or emotion control (not supported by Turbo). MeanFlow S3Gen (2 timesteps).
        chunk_size=25 ≈ 1 second of audio per chunk.
        """
        import time

        start_time = time.time()
        metrics = StreamingMetrics()

        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, norm_loudness=norm_loudness)
        else:
            assert self.conds is not None, "Please call prepare_conditionals() first or pass audio_prompt_path"

        text = punc_norm(text)
        text_tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_tokens = text_tokens.input_ids.to(self.device)

        total_audio = 0.0
        all_tokens = None
        last_chunk = None

        with torch.inference_mode():
            for token_chunk in self.t3.inference_turbo_stream(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                chunk_size=chunk_size,
            ):
                token_chunk = token_chunk[0]  # unbatch
                last_chunk = token_chunk

                audio, duration = self._process_token_chunk(
                    token_chunk, all_tokens, context_window, start_time, metrics, fade_duration
                )

                if audio is not None:
                    total_audio += duration
                    yield audio, metrics

                all_tokens = token_chunk if all_tokens is None else torch.cat([all_tokens, token_chunk], dim=-1)

            # Append silence tokens to the final chunk and decode remainder
            if last_chunk is not None:
                silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL], dtype=torch.long, device=self.device)
                final_tokens = torch.cat([last_chunk, silence])
                audio, duration = self._process_token_chunk(
                    final_tokens, all_tokens, context_window, start_time, metrics, fade_duration
                )
                if audio is not None:
                    total_audio += duration
                    yield audio, metrics

        metrics.total_generation_time = time.time() - start_time
        metrics.total_audio_duration = total_audio
        if total_audio > 0:
            metrics.rtf = metrics.total_generation_time / total_audio
