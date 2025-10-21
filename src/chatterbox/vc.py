# chatterbox/src/chatterbox/vc.py

from pathlib import Path

import librosa
import torch
import perth
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from typing import List, Union

from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen
from .models.s3gen.const import TOKEN_TO_WAV_RATIO


REPO_ID = "ResembleAI/chatterbox"


class ChatterboxVC:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR
    TOKEN_TO_WAV_RATIO = 960

    def __init__(
        self,
        s3gen: S3Gen,
        device: str,
        ref_dict: dict = None,
    ):
        self.sr = S3GEN_SR
        self.s3gen = s3gen
        self.device = device
        self.watermarker = perth.PerthImplicitWatermarker()
        if ref_dict is None:
            self.ref_dict = None
        else:
            self.ref_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in ref_dict.items()
            }

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxVC':
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ref_dict = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            states = torch.load(builtin_voice, map_location=map_location)
            ref_dict = states['gen']

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        return cls(s3gen, device, ref_dict=ref_dict)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxVC':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        for fpath in ["s3gen.safetensors", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def set_target_voice(self, wav_fpaths: Union[str, List[str]]):
        if isinstance(wav_fpaths, str):
            wav_fpaths = [wav_fpaths]
        
        s3gen_ref_wavs = []
        for fpath in wav_fpaths:
            s3gen_ref_wav, _ = librosa.load(fpath, sr=S3GEN_SR)
            s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
            s3gen_ref_wavs.append(torch.from_numpy(s3gen_ref_wav))
        
        s3gen_ref_batch = torch.nn.utils.rnn.pad_sequence(s3gen_ref_wavs, batch_first=True)
        self.ref_dict = self.s3gen.embed_ref(s3gen_ref_batch, S3GEN_SR, device=self.device)

    def generate(
        self,
        audio: Union[str, List[str]],
        target_voice_path: Union[str, List[str]] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        is_single_input = isinstance(audio, str)
        if is_single_input:
            audio = [audio]
        batch_size = len(audio)

        if target_voice_path:
            self.set_target_voice(target_voice_path)
        else:
            assert self.ref_dict is not None, "Please call `set_target_voice` first or specify `target_voice_path`"
        
        # Broadcast conditioning if a single prompt is used for a batch of inputs
        current_cond_bs = self.ref_dict['embedding'].size(0)
        if current_cond_bs == 1 and batch_size > 1:
            for k, v in self.ref_dict.items():
                if torch.is_tensor(v):
                    if k.endswith("_len"):
                        self.ref_dict[k] = v.expand(batch_size)
                    else:
                        self.ref_dict[k] = v.expand(batch_size, *v.shape[1:])
        elif current_cond_bs != batch_size and not (current_cond_bs == 1 and batch_size == 1):
            raise ValueError(f"Mismatch between number of source audios ({batch_size}) and target voice paths ({current_cond_bs})")

        with torch.inference_mode():
            audios_16k = []
            for a in audio:
                audio_16, _ = librosa.load(a, sr=S3_SR)
                audios_16k.append(torch.from_numpy(audio_16).float())

            audio_16_padded = torch.nn.utils.rnn.pad_sequence(audios_16k, batch_first=True).to(self.device)

            s3_tokens, s3_token_lens = self.s3gen.tokenizer(audio_16_padded)
            wavs, _ = self.s3gen.inference(
                speech_tokens=s3_tokens,
                speech_token_lens=s3_token_lens,
                ref_dict=self.ref_dict,
            )
            # Trim padding noise
            audio_lengths = s3_token_lens * TOKEN_TO_WAV_RATIO
            output_tensors = []
            for i, wav in enumerate(wavs):
                trimmed_wav = wav[:audio_lengths[i]].cpu().numpy()
                watermarked_wav = self.watermarker.apply_watermark(trimmed_wav, sample_rate=self.sr)
                output_tensors.append(torch.from_numpy(watermarked_wav).unsqueeze(0))

        if is_single_input:
            return output_tensors[0]
        return output_tensors