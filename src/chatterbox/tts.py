from dataclasses import dataclass
from pathlib import Path

import time
from typing import Generator, Tuple, Optional
import re

import librosa
import numpy as np
import torch
import perth
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond

from .models.t3.inference.alignment_stream_analyzer import AlignmentStreamAnalyzer
from .models.t3.inference.t3_hf_backend import T3HuggingfaceBackend



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


@dataclass
class StreamingMetrics:
    """Metrics for streaming TTS generation"""
    latency_to_first_chunk: Optional[float] = None
    rtf: Optional[float] = None
    total_generation_time: Optional[float] = None
    total_audio_duration: Optional[float] = None
    chunk_count: int = 0

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

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

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
        min_p=0.05,
        top_p=1.0,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
    ):
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        if cfg_weight > 0.0:
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
            
            speech_tokens = speech_tokens[speech_tokens < 6561]

            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    def _process_token_buffer(
        self,
        token_buffer,
        all_tokens_so_far,
        context_window,
    ):
        # Combine buffered chunks of tokens
        new_tokens = torch.cat(token_buffer, dim=-1)

        # Build tokens_to_process by including a context window
        if len(all_tokens_so_far) > 0:
            context_tokens = all_tokens_so_far[-context_window:] # In the case that all_tokens_so_far is less than context tokens, python slicing will return al of all_tokens_so_far
            tokens_to_process = torch.cat([context_tokens, new_tokens], dim=-1)
            context_length = len(context_tokens)
        else:
            tokens_to_process = new_tokens
            context_length = 0


        # Drop any invalid tokens and move to the correct device
        # TODO -> Check if the next two lines of code are redundant, we may only need one
        speech_tokens = tokens_to_process[tokens_to_process < 6561]

        speech_tokens = drop_invalid_tokens(speech_tokens)

        speech_tokens = speech_tokens.to(self.device)

        # NOTE -> I believe this exists since we're not processing all speech tokens so we might recieve no speech tokens within the streaming process
        if len(speech_tokens) == 0:
            return None, False

        # Run S3Gen inference to get a waveform (1 × T)
        wav, _ = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=self.conds.gen,
        )
        wav = wav.squeeze(0).detach().cpu().numpy()

        # If we have context tokens, crop out the samples corresponding to them
        # TODO -> this seems impercise considering different tokens may have different audio lengths. Something to consider.
        if context_length > 0:
            samples_per_token = len(wav) / len(speech_tokens)
            skip_samples = int(context_length * samples_per_token)
            audio_chunk = wav[skip_samples:]
        else:
            audio_chunk = wav
    
        # TODO -> This may also be necessary considering that we are processing chunks instead all speech tokens, however, it seems redundant due to the previous check at line 456.
        if len(audio_chunk) == 0:
            return None, False

        # Apply a short linear fade-in on the new chunk to smooth boundaries
        # TODO -> change to cross fading as opposed to a simple fade in
        # fade_samples = int(fade_duration * self.sr)
        # if fade_samples > 0:
        #     if fade_samples > len(audio_chunk):
        #         fade_samples = len(audio_chunk)
        #     fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=audio_chunk.dtype)
        #     audio_chunk[:fade_samples] *= fade_in

        # Compute audio duration and watermark
        audio_duration = len(audio_chunk) / self.sr
        # NOTE -> removing watermarking to improve speed, also since this will be used in a streaming context and not for distribution watermarking isn't as necessary.
        # watermarked_chunk = self.watermarker.apply_watermark(audio_chunk, sample_rate=self.sr)

        # NOTE -> Why convert back into a tensor? I'm removing this for now so that we can create a audio buffer that manages the audio as a numpy array.
        # audio_tensor = torch.from_numpy(watermarked_chunk).unsqueeze(0)

        #return audio_chunk, audio_duration, True
        return audio_chunk, True

    def setup_stream(
        self,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5
    ):
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)


    def generate_stream(
        self,
        text: str,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        chunk_size: int = 25,  # Tokens per chunk
        context_window = 50,
    ) -> Generator[Tuple[torch.Tensor, StreamingMetrics], None, None]:
        """
        Streaming version of generate that yields audio chunks as they are generated.
        
        Args:
            text: Input text to synthesize
            audio_prompt_path: Optional path to reference audio for voice cloning
            exaggeration: Emotion exaggeration factor
            cfg_weight: Classifier-free guidance weight
            temperature: Sampling temperature
            chunk_size: Number of speech tokens per chunk
            context_window: The context passed for each chunk
            fade_duration: Seconds to apply linear fade-in on each chunk
            
        Yields:
            Tuple of (audio_chunk, metrics) where audio_chunk is a torch.Tensor
            and metrics contains timing information
        """
        # chunk text by sentence to avoid token limit
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences]

        # total_audio_length = 0.0
    
        for s in sentences:
            # Norm and tokenize text
            sentence = punc_norm(s)
            text_tokens = self.tokenizer.text_to_tokens(sentence).to(self.device)
            
            # While cfg_weight is not essential to TTS generation it improves quality of the output and adherence to conditions. For the purposes of this repository we will require it to be set to a non-zero value.
            if not cfg_weight > 0.0:
                raise ValueError("cfg_weight must be greater than zero")
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0) # Need two seqs for CFG.

            sot = self.t3.hp.start_text_token
            eot = self.t3.hp.stop_text_token
            text_tokens = F.pad(text_tokens, (1, 0), value=sot)
            text_tokens = F.pad(text_tokens, (0, 1), value=eot)

            # all_tokens_processed = []  # Keep track of all tokens processed so far

            with torch.inference_mode():
                # Stream speech tokens
                for token_chunk in self.t3.inference_stream(
                    t3_cond=self.conds.t3,
                    text_tokens=text_tokens,
                    max_new_tokens=1000,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    chunk_size=chunk_size,
                    repetition_penalty=repetition_penalty,
                    min_p=min_p,
                    top_p=top_p,
                ):
                    yield token_chunk
                    # Extract only the conditional batch
                    # token_chunk = token_chunk[0]

                    # # Process each chunk immediately
                    # audio_tensor, audio_duration, success = self._process_token_buffer(
                    #     [token_chunk], all_tokens_processed, context_window
                    # )

                    # if success:
                    #     total_audio_length += audio_duration
                    #     yield audio_tensor

                    # # Update all_tokens_processed with the new tokens
                    # # TODO -> all_tokens_processed
                    # if len(all_tokens_processed) == 0:
                    #     all_tokens_processed = token_chunk
                    # else:
                    #     all_tokens_processed = torch.cat([all_tokens_processed, token_chunk], dim=-1)

        #print(f"Total audio duration: {total_audio_length}")
    