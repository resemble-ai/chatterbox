from dataclasses import dataclass
from pathlib import Path
import os

import torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

from .models.t3 import T3
from .models.t3.modules.t3_config import T3Config
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import MTLTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond
from .tensor_utils import (
    load_t3_state_dict_safe,
    load_s3gen_safe,
    load_voice_encoder_safe, 
    load_conditionals_safe
)
from .shared_utils import (
    check_exaggeration_update_needed,
    check_mps_availability,
    get_map_location,
    validate_audio_file,
    load_and_preprocess_audio,
    drop_bad_tokens,
    prepare_text_tokens,
    punc_norm,
    validate_exaggeration,
    check_exaggeration_update_needed,
    validate_text_input,
    validate_language_id,
    validate_float_parameter,
    validate_audio_prompt_path,
    smart_text_splitter,
    estimate_token_count,
    concatenate_audio_tensors
)


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

    @classmethod
    def get_supported_languages(cls):
        """Return dictionary of supported language codes and names."""
        return SUPPORTED_LANGUAGES.copy()

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(ckpt_dir)

        # Load voice encoder
        ve = load_voice_encoder_safe(ckpt_dir, device, is_multilingual=True)

        # Load T3 model with multilingual config
        t3 = T3(T3Config.multilingual())
        load_t3_state_dict_safe(t3, ckpt_dir / "t3_23lang.safetensors", device)

        # Load S3Gen model
        s3gen = load_s3gen_safe(ckpt_dir, device, is_multilingual=True)

        # Load tokenizer
        tokenizer = MTLTokenizer(str(ckpt_dir / "mtl_tokenizer.json"))

        # Load conditionals if they exist
        conds = load_conditionals_safe(ckpt_dir, device, is_multilingual=True)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device: torch.device) -> 'ChatterboxMultilingualTTS':
        # Use shared MPS checking utility
        device = check_mps_availability(device)

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
        """
        # Use shared validation utilities
        validate_audio_file(wav_fpath)
        exaggeration = validate_exaggeration(exaggeration)
        
        # Use shared audio preprocessing
        s3gen_ref_wav, ref_16k_wav_tensor = load_and_preprocess_audio(wav_fpath, self.device)

        # Slice as tensor
        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)
        
        # Speech cond prompt tokens
        t3_cond_prompt_tokens = None
        plen = getattr(self.t3.hp, 'speech_cond_prompt_len', 30)
        if plen:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav_tensor[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding - use tensor-native method, keep on device
        ve_embed = self.ve.embeds_from_wavs_tensor([ref_16k_wav_tensor], sample_rate=S3_SR)
        ve_embed = ve_embed.mean(dim=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1, dtype=ve_embed.dtype),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def set_conditionals(self, conds: Conditionals):
        """Set conditionals for the TTS model."""
        self.conds = conds

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
        # Enhanced input validation using shared utilities
        text = validate_text_input(text)
        language_id = validate_language_id(language_id, SUPPORTED_LANGUAGES)
        
        # Validate numerical parameters using shared validation
        exaggeration = validate_float_parameter(exaggeration, "exaggeration", 0.0, 2.0)
        cfg_weight = validate_float_parameter(cfg_weight, "cfg_weight", 0.0, 1.0)
        temperature = validate_float_parameter(temperature, "temperature", allow_zero=False)
        min_p = validate_float_parameter(min_p, "min_p", 0.0, 1.0)
        top_p = validate_float_parameter(top_p, "top_p", 0.0, 1.0)
        repetition_penalty = validate_float_parameter(repetition_penalty, "repetition_penalty", allow_zero=False)
        
        # Handle audio prompt path
        if audio_prompt_path:
            validate_audio_prompt_path(audio_prompt_path)
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            if self.conds is None:
                raise ValueError("Please `prepare_conditionals` first or specify `audio_prompt_path`")

        # Update exaggeration if needed using shared utility
        if self.conds is not None:
            needs_update, new_emotion_tensor = check_exaggeration_update_needed(
                self.conds.t3.emotion_adv, exaggeration, self.device
            )
            if needs_update:
                _cond: T3Cond = self.conds.t3
                self.conds.t3 = T3Cond(
                    speaker_emb=_cond.speaker_emb,
                    cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                    emotion_adv=new_emotion_tensor,
                ).to(device=self.device, dtype=self.conds.t3.speaker_emb.dtype)

        # Normalize and check if text needs chunking
        text = punc_norm(text, multilingual=True)
        
        # Check if text needs to be chunked based on token count
        estimated_tokens = estimate_token_count(text, self.tokenizer, language_id)
        # Set chunk limit based on cache constraints: max_cache_len - max_new_tokens
        # This prevents the "max_cache_len too small" warning and ensures optimal performance
        max_chunk_tokens = max(200, max_cache_len - max_new_tokens - 75)  # 75 token safety margin

        if estimated_tokens <= max_chunk_tokens:
            # Text is small enough - process normally without chunking
            return self._generate_single_chunk(
                text, language_id, cfg_weight, max_new_tokens, temperature,
                max_cache_len, repetition_penalty, min_p, top_p, t3_params
            )
        else:
            # Text is too large - split into chunks and process separately
            #print(f"Text too large ({estimated_tokens} tokens), splitting into chunks...")
            #print(f"Using cache-aware chunk limit: {max_chunk_tokens} tokens (cache_len={max_cache_len}, max_new={max_new_tokens})")
            text_chunks = smart_text_splitter(text, max_chunk_tokens, self.tokenizer, language_id)
            #print(f"Split into {len(text_chunks)} chunks")
            
            # Store original conditionals for reuse
            original_conds = self.conds.clone()
            
            audio_chunks = []
            for i, chunk in enumerate(text_chunks):
                #print(f"Processing chunk {i+1}/{len(text_chunks)}")
                
                # Reset conditionals for each chunk to ensure consistency
                self.conds = original_conds.clone()
                
                chunk_audio = self._generate_single_chunk(
                    chunk, language_id, cfg_weight, max_new_tokens, temperature,
                    max_cache_len, repetition_penalty, min_p, top_p, t3_params
                )
                audio_chunks.append(chunk_audio)
            
            # Concatenate all audio chunks with brief silence between them
            return concatenate_audio_tensors(audio_chunks, silence_duration=0.1, sample_rate=self.sr)

    def _generate_single_chunk(
        self, 
        text, 
        language_id,
        cfg_weight, 
        max_new_tokens, 
        temperature, 
        max_cache_len, 
        repetition_penalty, 
        min_p, 
        top_p, 
        t3_params
    ):
        """Generate audio for a single text chunk."""
        text_tokens = self.tokenizer.text_to_tokens(text, language_id=language_id).to(self.device)

        # Use shared text token preparation
        text_tokens = prepare_text_tokens(
            text_tokens, 
            self.t3.hp.start_text_token, 
            self.t3.hp.stop_text_token, 
            cfg_weight
        )

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

            def speech_to_wav(speech_tokens):
                # Extract only the conditional batch.
                speech_tokens = speech_tokens[0]

                speech_tokens = drop_invalid_tokens(speech_tokens)
                speech_tokens = drop_bad_tokens(speech_tokens)
                
                if len(speech_tokens) == 0:
                    raise RuntimeError("No valid speech tokens generated")
                
                wav, _ = self.s3gen.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=self.conds.gen,
                )

                return wav
                
            return speech_to_wav(speech_tokens)
