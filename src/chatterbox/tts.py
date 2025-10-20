# chatterbox/tts.py

from dataclasses import dataclass
from pathlib import Path

import librosa
import torch
import perth
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from typing import Union, List

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.s3gen.const import TOKEN_TO_WAV_RATIO
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/chatterbox"


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


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
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
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTTS':
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

        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(
            str(ckpt_dir / "tokenizer.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTTS':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def prepare_conditionals(self, wav_fpaths: Union[str, List[str]], exaggeration=0.5):
        if isinstance(wav_fpaths, str):
            wav_fpaths = [wav_fpaths]
        
        ## Load reference wav
        s3gen_ref_wavs, ref_16k_wavs_np = [], []
        for fpath in wav_fpaths:
            s3gen_wav, _ = librosa.load(fpath, sr=S3GEN_SR)
            s3gen_wav_tensor = torch.from_numpy(s3gen_wav[:self.DEC_COND_LEN])
            s3gen_ref_wavs.append(s3gen_wav_tensor)

            ref_16k_wav, _ = librosa.load(fpath, sr=S3_SR)
            ref_16k_wavs_np.append(ref_16k_wav)

        s3gen_ref_batch = torch.nn.utils.rnn.pad_sequence(s3gen_ref_wavs, batch_first=True)
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_batch, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            ref_16k_prompts = [wav[:self.ENC_COND_LEN] for wav in ref_16k_wavs_np]
            t3_cond_prompt_tokens, _ = s3_tokzr.forward(ref_16k_prompts, max_len=plen)
            t3_cond_prompt_tokens = t3_cond_prompt_tokens.to(self.device)
        else:
            t3_cond_prompt_tokens = None

        # Voice-encoder speaker embedding
        ve_embeds = self.ve.embeds_from_wavs(ref_16k_wavs_np, sample_rate=S3_SR)
        ve_embed = torch.from_numpy(ve_embeds).unsqueeze(1).to(self.device)

        batch_size = len(wav_fpaths)
        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(batch_size, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text: Union[str, List[str]],
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        audio_prompt_path: Union[str, List[str]] = None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        num_return_sequences=1,
    ) -> Union[torch.Tensor, List[torch.Tensor], List[List[torch.Tensor]]]:
        is_single_input = isinstance(text, str)
        if is_single_input:
            text = [text]
        batch_size = len(text)

        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Broadcast conditioning if a single prompt is used for a batch of texts
        current_cond_bs = self.conds.t3.speaker_emb.size(0)
        if current_cond_bs == 1 and batch_size > 1:
            t3c = self.conds.t3
            t3c.speaker_emb = t3c.speaker_emb.expand(batch_size, -1, -1)
            if t3c.cond_prompt_speech_tokens is not None:
                t3c.cond_prompt_speech_tokens = t3c.cond_prompt_speech_tokens.expand(batch_size, -1)
            if t3c.emotion_adv is not None:
                t3c.emotion_adv = t3c.emotion_adv.expand(batch_size, -1, -1)

            gend = self.conds.gen
            for k, v in gend.items():
                if torch.is_tensor(v):
                    if k.endswith("_len"):
                        gend[k] = v.expand(batch_size)
                    else:
                        gend[k] = v.expand(batch_size, *v.shape[1:])
        elif current_cond_bs != batch_size and not (current_cond_bs == 1 and batch_size == 1):
             raise ValueError(f"Mismatch between number of texts ({batch_size}) and audio prompts ({current_cond_bs})")

        # Update exaggeration if needed (applied to the whole batch)
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            self.conds.t3.emotion_adv = exaggeration * torch.ones(batch_size, 1, 1, device=self.device)

        # Norm and tokenize text
        texts = [punc_norm(t) for t in text]
        tokenized_texts = [self.tokenizer.text_to_tokens(t).squeeze(0) for t in texts]
        text_tokens = torch.nn.utils.rnn.pad_sequence(tokenized_texts, batch_first=True, padding_value=0).to(self.device)

        # --- Start: Logic for num_return_sequences and CFG ---
        t3_cond = self.conds.t3
        gen_cond = self.conds.gen

        # Expand inputs for num_return_sequences
        if num_return_sequences > 1:
            text_tokens = text_tokens.repeat_interleave(num_return_sequences, dim=0)
            t3_cond = T3Cond(**{k: v.repeat_interleave(num_return_sequences, dim=0) if torch.is_tensor(v) else v for k, v in t3_cond.__dict__.items()})
            gen_cond = {k: v.repeat_interleave(num_return_sequences, dim=0) if torch.is_tensor(v) else v for k, v in gen_cond.items()}

        if cfg_weight > 0.0:
            # Duplicate text tokens and conditioning tensors for CFG
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
            t3_cond = T3Cond(**{k: torch.cat([v, v], dim=0) if torch.is_tensor(v) else v for k, v in t3_cond.__dict__.items()})
        # --- End: Logic for num_return_sequences and CFG ---


        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            # T3 generates a list of variable-length token sequences
            speech_tokens_list = self.t3.inference(
                t3_cond=t3_cond,
                text_tokens=text_tokens,
                max_new_tokens=1000,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )

            # Pad for filtering, filter, and pad again for S3Gen
            speech_tokens_padded = torch.nn.utils.rnn.pad_sequence(speech_tokens_list, batch_first=True, padding_value=self.t3.hp.stop_speech_token)
            clean_tokens_list = drop_invalid_tokens(speech_tokens_padded)
            s3gen_tokens_padded = torch.nn.utils.rnn.pad_sequence(clean_tokens_list, batch_first=True, padding_value=0)
            s3gen_token_lens = torch.tensor([len(t) for t in clean_tokens_list], device=self.device)

            wavs, _ = self.s3gen.inference(
                speech_tokens=s3gen_tokens_padded.to(self.device),
                speech_token_lens=s3gen_token_lens,
                ref_dict=gen_cond,
            )

            # Trim padding noise
            audio_lengths = s3gen_token_lens * TOKEN_TO_WAV_RATIO
            output_tensors = []
            for i, wav in enumerate(wavs):
                trimmed_wav = wav[:audio_lengths[i]].cpu().numpy()
                watermarked_wav = self.watermarker.apply_watermark(trimmed_wav, sample_rate=self.sr)
                output_tensors.append(torch.from_numpy(watermarked_wav).unsqueeze(0))

        if num_return_sequences > 1:
            # Group the flat list of outputs into a list of lists
            grouped_outputs = [output_tensors[i:i + num_return_sequences] for i in range(0, len(output_tensors), num_return_sequences)]
            if is_single_input:
                return grouped_outputs[0]
            return grouped_outputs

        if is_single_input:
            return output_tensors[0]
        return output_tensors