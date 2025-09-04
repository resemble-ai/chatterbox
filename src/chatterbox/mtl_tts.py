from dataclasses import dataclass
from pathlib import Path
import os
import warnings

import torch
import perth
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as taF
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

from .models.t3 import T3
from .models.t3.modules.t3_config import T3Config
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import MTLTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/chatterbox"

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
        (""", "\""),
        (""", "\""),
        ("'", "'"),
        ("'", "'"),
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
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(
            torch.load(ckpt_dir / "ve.pt", weights_only=True)
        )
        ve.to(device).eval()

        t3 = T3(T3Config.multilingual())
        t3_state = load_file(ckpt_dir / "t3_23lang.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            torch.load(ckpt_dir / "s3gen.pt", weights_only=True)
        )
        s3gen.to(device).eval()

        tokenizer = MTLTokenizer(
            str(ckpt_dir / "mtl_tokenizer.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device: torch.device) -> 'ChatterboxMultilingualTTS':
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
                allow_patterns=["ve.pt", "t3_23lang.safetensors", "s3gen.pt", "mtl_tokenizer.json", "conds.pt", "Cangjie5_TC.json"],
                token=os.getenv("HF_TOKEN"),
            )
        )
        return cls.from_local(ckpt_dir, device)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        """
        Prepare conditioning data from a reference audio file.
        
        Args:
            wav_fpath: Path to the reference audio file
            exaggeration: Float between 0.0 and 2.0 for emotion control
            
        Raises:
            FileNotFoundError: If the audio file doesn't exist
            ValueError: If the audio file is invalid or parameters are out of range
            RuntimeError: If audio processing fails
        """
        # Validate inputs
        if not isinstance(wav_fpath, (str, Path)):
            raise TypeError("wav_fpath must be a string or Path object")
        
        wav_path = Path(wav_fpath)
        if not wav_path.exists():
            raise FileNotFoundError(f"Reference audio file not found: {wav_fpath}")
        
        if not wav_path.is_file():
            raise ValueError(f"Reference audio path is not a file: {wav_fpath}")
        
        try:
            exaggeration = float(exaggeration)
            if not (0.0 <= exaggeration <= 2.0):
                raise ValueError("exaggeration must be between 0.0 and 2.0")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid exaggeration value: {e}")
        
        try:
            # Load reference wav with enhanced error handling
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                try:
                    s3gen_ref_wav, _sr = torchaudio.load(wav_fpath)
                except Exception as e:
                    raise RuntimeError(f"Failed to load audio file '{wav_fpath}': {e}")
            
            # Validate audio data
            if s3gen_ref_wav.numel() == 0:
                raise ValueError("Audio file is empty or contains no valid audio data")
            
            if _sr <= 0:
                raise ValueError(f"Invalid sample rate: {_sr}")
            
            # Check audio duration (minimum 0.1 seconds)
            min_duration = 0.1
            duration = s3gen_ref_wav.shape[-1] / _sr
            if duration < min_duration:
                raise ValueError(f"Audio too short: {duration:.2f}s (minimum {min_duration}s required)")
            
            # Resample if necessary
            if _sr != S3GEN_SR:
                try:
                    s3gen_ref_wav = taF.resample(s3gen_ref_wav, _sr, S3GEN_SR)
                except Exception as e:
                    raise RuntimeError(f"Failed to resample audio from {_sr}Hz to {S3GEN_SR}Hz: {e}")
            
            # Ensure we have mono audio (1D tensor)
            if s3gen_ref_wav.dim() > 1:
                s3gen_ref_wav = s3gen_ref_wav.mean(dim=0)  # Convert to mono by averaging channels
            
            # Move to device early to avoid unnecessary transfers
            s3gen_ref_wav = s3gen_ref_wav.to(self.device)
            
            # Resample to 16kHz directly from tensor
            try:
                ref_16k_wav_tensor = taF.resample(s3gen_ref_wav.unsqueeze(0), S3GEN_SR, S3_SR).squeeze(0)
            except Exception as e:
                raise RuntimeError(f"Failed to resample to 16kHz: {e}")

            # Slice as tensor
            s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
            
            try:
                s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)
            except Exception as e:
                raise RuntimeError(f"Failed to generate S3Gen reference embeddings: {e}")
            
            # Speech cond prompt tokens
            t3_cond_prompt_tokens = None
            if plen := self.t3.hp.speech_cond_prompt_len:
                try:
                    s3_tokzr = self.s3gen.tokenizer
                    t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav_tensor[:self.ENC_COND_LEN]], max_len=plen)
                    t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)
                except Exception as e:
                    raise RuntimeError(f"Failed to generate speech conditioning tokens: {e}")

            # Voice-encoder speaker embedding - use tensor-native method, keep on device
            try:
                ve_embed = self.ve.embeds_from_wavs_tensor([ref_16k_wav_tensor], sample_rate=S3_SR)
                ve_embed = ve_embed.mean(dim=0, keepdim=True).to(self.device)
            except Exception as e:
                raise RuntimeError(f"Failed to generate voice encoder embeddings: {e}")

            try:
                t3_cond = T3Cond(
                    speaker_emb=ve_embed,
                    cond_prompt_speech_tokens=t3_cond_prompt_tokens,
                    emotion_adv=exaggeration * torch.ones(1, 1, 1),
                ).to(device=self.device)
                self.conds = Conditionals(t3_cond, s3gen_ref_dict)
            except Exception as e:
                raise RuntimeError(f"Failed to create conditioning objects: {e}")
                
        except Exception as e:
            # Re-raise our custom exceptions as-is, wrap others in RuntimeError
            if isinstance(e, (FileNotFoundError, ValueError, RuntimeError)):
                raise
            else:
                raise RuntimeError(f"Unexpected error in prepare_conditionals: {e}")

    def generate(
        self,
        text,
        language_id,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        # cache optimization params
        max_new_tokens=750, 
        max_cache_len=1050, # Affects the T3 speed, hence important
        # t3 sampling params
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        t3_params={},
    ):
        # Enhanced input validation
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        if text.strip() == "":
            raise ValueError("Text cannot be empty or whitespace only")
        
        # Validate language_id
        if language_id is None:
            raise ValueError("language_id is required for multilingual TTS. Use one of: " + ", ".join(SUPPORTED_LANGUAGES.keys()))
        
        if not isinstance(language_id, str):
            raise TypeError(f"language_id must be a string, got {type(language_id)}")
        
        if language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )
        
        # Validate numerical parameters
        try:
            exaggeration = float(exaggeration)
            if not (0.0 <= exaggeration <= 2.0):
                raise ValueError("exaggeration must be between 0.0 and 2.0")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid exaggeration value: {e}")
        
        try:
            cfg_weight = float(cfg_weight)
            if not (0.0 <= cfg_weight <= 1.0):
                raise ValueError("cfg_weight must be between 0.0 and 1.0")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid cfg_weight value: {e}")
        
        try:
            temperature = float(temperature)
            if temperature <= 0.0:
                raise ValueError("temperature must be positive")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid temperature value: {e}")
        
        # Validate sampling parameters
        try:
            min_p = float(min_p)
            if not (0.0 <= min_p <= 1.0):
                raise ValueError("min_p must be between 0.0 and 1.0")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid min_p value: {e}")
        
        try:
            top_p = float(top_p)
            if not (0.0 <= top_p <= 1.0):
                raise ValueError("top_p must be between 0.0 and 1.0")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid top_p value: {e}")
        
        try:
            repetition_penalty = float(repetition_penalty)
            if repetition_penalty <= 0.0:
                raise ValueError("repetition_penalty must be positive")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid repetition_penalty value: {e}")
        
        # Handle audio prompt path with better error messages
        if audio_prompt_path:
            if not isinstance(audio_prompt_path, (str, Path)):
                raise TypeError("audio_prompt_path must be a string or Path object")
            
            audio_path = Path(audio_prompt_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio prompt file not found: {audio_prompt_path}")
            
            if not audio_path.is_file():
                raise ValueError(f"Audio prompt path is not a file: {audio_prompt_path}")
            
            # Check file extension
            valid_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
            if audio_path.suffix.lower() not in valid_extensions:
                raise ValueError(f"Unsupported audio format: {audio_path.suffix}. Supported: {', '.join(valid_extensions)}")
            
            try:
                self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
            except Exception as e:
                raise RuntimeError(f"Failed to prepare conditionals from audio file: {e}")
        else:
            if self.conds is None:
                raise ValueError("Please `prepare_conditionals` first or specify `audio_prompt_path`")

        # Update exaggeration if needed
        try:
            if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
                _cond: T3Cond = self.conds.t3
                self.conds.t3 = T3Cond(
                    speaker_emb=_cond.speaker_emb,
                    cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                    emotion_adv=exaggeration * torch.ones(1, 1, 1),
                ).to(device=self.device, dtype=self.conds.t3.speaker_emb.dtype)
        except Exception as e:
            raise RuntimeError(f"Failed to update exaggeration: {e}")

        # Normalize and tokenize text with enhanced error handling
        try:
            text = punc_norm(text)
        except Exception as e:
            raise RuntimeError(f"Failed to normalize text: {e}")
        
        try:
            text_tokens = self.tokenizer.text_to_tokens(text, language_id=language_id.lower()).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to tokenize text for language '{language_id}': {e}")

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        try:
            with torch.inference_mode():
                speech_tokens = self.t3.inference(
                    t3_cond=self.conds.t3,
                    text_tokens=text_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    max_cache_len=max_cache_len,
                    repetition_penalty=repetition_penalty,
                    min_p=min_p,
                    top_p=top_p,
                    **t3_params,
                )
        except Exception as e:
            raise RuntimeError(f"T3 inference failed: {e}")
        
        try:
            def speech_to_wav(speech_tokens):
                # Extract only the conditional batch.
                speech_tokens = speech_tokens[0]

                # TODO: output becomes 1D
                speech_tokens = drop_invalid_tokens(speech_tokens)
                
                def drop_bad_tokens(tokens):
                    # Use torch.where instead of boolean indexing to avoid sync
                    mask = tokens < 6561
                    # Count valid tokens without transferring to CPU
                    valid_count = torch.sum(mask).item()
                    # Create output tensor of the right size
                    result = torch.zeros(valid_count, dtype=tokens.dtype, device=tokens.device)
                    # Use torch.masked_select which is more CUDA-friendly
                    result = torch.masked_select(tokens, mask)
                    return result

                # speech_tokens = speech_tokens[speech_tokens < 6561]
                speech_tokens = drop_bad_tokens(speech_tokens)
                
                if len(speech_tokens) == 0:
                    raise RuntimeError("No valid speech tokens generated")
                
                wav, _ = self.s3gen.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=self.conds.gen,
                )

                # Apply watermarking
                wav = wav.squeeze(0).detach().cpu().numpy()
                watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
                return torch.from_numpy(watermarked_wav).unsqueeze(0)
                
            return speech_to_wav(speech_tokens)
        except Exception as e:
            raise RuntimeError(f"Audio generation failed: {e}")
