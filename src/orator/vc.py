from dataclasses import dataclass
from pathlib import Path

import librosa
import torch
import torch.nn.functional as F

from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen


class OratorVC:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        s3gen: S3Gen,
        device: str,
        ref_dict=None,
    ):
        self.sr = S3GEN_SR
        self.s3gen = s3gen
        self.device = device
        if ref_dict is None:
            self.ref_dict = None
        else:
            ref_dict: dict
            self.ref_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in ref_dict.items()
            }

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'OratorVC':
        ckpt_dir = Path(ckpt_dir)
        ref_dict = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            states = torch.load(builtin_voice)
            ref_dict = states['gen']

        s3gen = S3Gen()
        s3gen.load_state_dict(
            torch.load(ckpt_dir / "s3gen.pt")
        )
        s3gen.to(device).eval()

        return cls(s3gen, device, ref_dict=ref_dict)

    @classmethod
    def from_pretrained(cls, model_name, device):
        """TODO HF?"""
        pass

    def set_target_voice(self, wav_fpath):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        self.ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

    def generate(
        self,
        audio,
        target_voice_path=None,
    ):
        if target_voice_path:
            self.set_target_voice(target_voice_path)
        else:
            assert self.ref_dict is not None, "Please `prepare_conditionals` first or specify `target_voice_path`"

        with torch.inference_mode():
            if isinstance(audio, str):
                import torchaudio as ta
                audio_16, _ = librosa.load(audio, sr=S3_SR)
                audio_16 = torch.from_numpy(audio_16).float().to(self.device)[None, ]
            else:
                raise NotImplementedError()

            s3_tokens, _ = self.s3gen.tokenizer(audio_16)
            wav, _ = self.s3gen.inference(
                speech_tokens=s3_tokens,
                ref_dict=self.ref_dict,
            )
        return wav.detach().cpu()
