import os
import math
from dataclasses import dataclass
from pathlib import Path

import librosa
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
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTurboTTS':
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
        ve.to(device).eval()

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
        t3.to(device).eval()

        s3gen = S3Gen(meanflow=True)
        weights = load_file(ckpt_dir / "s3gen_meanflow.safetensors")
        s3gen.load_state_dict(
            weights, strict=True
        )
        s3gen.to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if len(tokenizer) != 50276:
            print(f"WARNING: Tokenizer len {len(tokenizer)} != 50276")

        conds = None
        builtin_voice = ckpt_dir / "conds.pt"
        if builtin_voice.exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTurboTTS':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        local_path = snapshot_download(
            repo_id=REPO_ID,
            token=os.getenv("HF_TOKEN") or True,
            # Optional: Filter to download only what you need
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]
        )

        return cls.from_local(local_path, device)

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
