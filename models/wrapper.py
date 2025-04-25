from dataclasses import dataclass
from pathlib import Path

import librosa
import torch
import torch.nn.functional as F

from models.t3 import T3
from models.s3tokenizer import S3_SR, drop_invalid_tokens
from models.s3gen import S3GEN_SR, S3Gen
from models.tokenizers import EnTokenizer
from models.voice_encoder import VoiceEncoder
from models.t3.modules.cond_enc import T3Cond


# TODO: but emotion_adv should by fluid.
@dataclass(frozen=True)
class Conditionals:
    """Conditionals for T3 and S3Gen"""
    lm: T3Cond
    voc: dict


class Orator:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str
    ):
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device

        # TODO: maybe load a default voice?
        self.t3_cond = None

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'Orator':
        ckpt_dir = Path(ckpt_dir)

        ve = VoiceEncoder()
        ve.load_state_dict(
            torch.load("checkpoints/ve.pt")
        )
        ve.to(device).eval()

        t3 = T3()
        t3.load_state_dict(
            torch.load(ckpt_dir / "t3.pt")
        )
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            torch.load(ckpt_dir / "s3gen.pt")
        )
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(
            str(ckpt_dir / "tokenizer.json")
        )
        return cls(t3, s3gen, ve, tokenizer, device)

    @classmethod
    def from_pretrained(cls, model_name, device):
        """HF?"""
        pass

    def prepare_conditionals(self, wav_fpath, emotion_adv=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        s3_ref_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([s3_ref_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([s3_ref_wav], sample_rate=S3GEN_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=emotion_adv * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        audio_prompt_path=None,
        emotion_adv=0.5
    ):
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, emotion_adv=emotion_adv)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"


        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.lm,
                text_tokens=text_tokens,
                max_new_tokens=1000,  # TODO: use the value in config
            )

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.voc,
            )

        return wav.detach().cpu()
