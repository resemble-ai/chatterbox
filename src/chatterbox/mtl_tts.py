from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import numpy as np
import logging as _mtl_logging

import librosa
import torch
import perth
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download
import psutil
from .models.t3 import T3
from .models.t3.modules.t3_config import T3Config
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import MTLTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond
from .models.utils import clear_device_memory

# Shared generation utilities
from .generation_utils import (
    SPACY_AVAILABLE,
    split_into_sentences,
    get_adaptive_chunks,
    split_text_intelligently,
    crossfade_chunks,
    estimate_max_tokens,
    print_generation_plan,
    print_chunk_generating,
    print_chunk_completed,
    print_generation_complete,
    print_crossfading,
)


def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


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

# Module-level logger for debugging
_mtl_logger = _mtl_logging.getLogger(__name__)


def punc_norm(text: str, debug: bool = True) -> str:
    """
    Quick cleanup func for punctuation from LLMs or
    containing chars not seen often in the dataset
    """
    original_text = text

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
        ("“", '"'),
        ("”", '"'),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",", "、", "，", "。", "？", "！"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    # DEBUG: Log punc_norm changes
    if debug and original_text != text:
        _mtl_logger.info(f"[punc_norm DEBUG] Input:  {repr(original_text[:200])}")
        _mtl_logger.info(f"[punc_norm DEBUG] Output: {repr(text[:200])}")

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
        arg_dict = dict(t3=self.t3.__dict__, gen=self.gen)
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs["t3"]), kwargs["gen"])


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
    def from_local(
        cls, ckpt_dir, device, t3_config=None
    ) -> "ChatterboxMultilingualTTS":
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device("cpu")
        else:
            map_location = None

        before_ve = get_memory_mb()
        print(f"before_ve: {before_ve:.1f}")
        ve = VoiceEncoder()
        ve.load_state_dict(
            torch.load(ckpt_dir / "ve.pt", weights_only=True, map_location=map_location)
        )
        ve.to(device).eval()

        before_t3 = get_memory_mb()
        print(f"before_t3: {before_t3:.1f}")
        # Use provided config or create default multilingual config
        if t3_config is None:
            t3_config = T3Config.multilingual()
        t3 = T3(t3_config)
        t3_state = load_safetensors(ckpt_dir / "t3_mtl23ls_v2.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        before_s3_gen = get_memory_mb()
        print(f"before_s3_gen: {before_s3_gen:.1f}")
        s3gen = S3Gen()
        s3gen.load_state_dict(
            torch.load(
                ckpt_dir / "s3gen.pt", weights_only=True, map_location=map_location
            )
        )
        s3gen.to(device).eval()
        before_tokenizer = get_memory_mb()
        print(f"before_tokenizer: {before_tokenizer:.1f}")
        tokenizer = MTLTokenizer(str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"))

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(
                device
            )
        after_from_local_load = get_memory_mb()
        print(f"after_from_local_load: {after_from_local_load:.1f}")
        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(
        cls, device: torch.device, t3_config=None
    ) -> "ChatterboxMultilingualTTS":
        ckpt_dir = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main",
                allow_patterns=[
                    "ve.pt",
                    "t3_mtl23ls_v2.safetensors",
                    "s3gen.pt",
                    "grapheme_mtl_merged_expanded_v1.json",
                    "conds.pt",
                    "Cangjie5_TC.json",
                ],
                token=os.getenv("HF_TOKEN"),
            )
        )
        return cls.from_local(ckpt_dir, device, t3_config=t3_config)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[: self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(
            s3gen_ref_wav, S3GEN_SR, device=self.device
        )

        # Speech cond prompt tokens
        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            # Limit audio length more aggressively for memory safety
            # Increased from 3 to 6 seconds for better conditioning
            safe_audio_len = min(len(ref_16k_wav), 6 * S3_SR)
            limited_audio = ref_16k_wav[:safe_audio_len]

            # Add memory cleanup before tokenization
            clear_device_memory()

            # Use smaller max_len to be extra safe
            safe_max_len = min(plen, 150)  # 150 tokens for better quality
            t3_cond_prompt_tokens, _ = s3_tokzr.forward(
                [limited_audio], max_len=safe_max_len
            )
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(
                self.device
            )

            # More memory cleanup after tokenization
            clear_device_memory()

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(
            self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR)
        )
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
        show_progress: bool = True,
    ):
        """
        Generate speech from text in a single pass.

        Use this method for short to medium-length text (up to ~50 words).
        For longer text, use `generate_long()` which handles memory more efficiently.

        Args:
            text: Input text to synthesize
            language_id: Language code (e.g., 'en', 'es', 'ja', 'zh'). See SUPPORTED_LANGUAGES for full list.
            audio_prompt_path: Path to reference audio file for voice cloning. If None, uses previously
                             prepared conditionals via `prepare_conditionals()`.
            exaggeration: Voice exaggeration/expressiveness level (0.0-1.0). Higher values produce more
                        expressive speech. Default: 0.5
            cfg_weight: Classifier-free guidance weight (0.0-1.0). Higher values follow conditioning more
                      closely. Default: 0.5. Set to 0 to disable CFG (not recommended).
            temperature: Sampling temperature for token generation (0.0-2.0). Higher values increase
                       randomness. Default: 0.8
            repetition_penalty: Penalty for repeating tokens (1.0-3.0). Higher values reduce repetition.
                              Default: 2.0. Try lowering to 1.2-1.5 for more natural speech.
            min_p: Minimum probability threshold for sampling. Tokens with probability below this are filtered.
                  Default: 0.05
            top_p: Nucleus sampling threshold. Only tokens with cumulative probability up to this value
                  are considered. Default: 1.0 (no filtering).
            show_progress: Whether to show token-level progress bar (default True)

        Returns:
            torch.Tensor: Generated audio waveform with shape (1, num_samples) at 24kHz sample rate.

        Raises:
            ValueError: If language_id is not in SUPPORTED_LANGUAGES.
            AssertionError: If audio_prompt_path is None and conditionals haven't been prepared.

        Example:
            >>> model = ChatterboxMultilingualTTS.from_pretrained()
            >>> wav = model.generate(
            ...     "Hello, how are you?",
            ...     language_id="en",
            ...     audio_prompt_path="reference.wav",
            ...     exaggeration=0.3,
            ...     cfg_weight=0.5
            ... )
            >>> torchaudio.save("output.wav", wav, model.sr)
        """
        import time as _time

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
            assert (
                self.conds is not None
            ), "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm text
        text = punc_norm(text)

        # Split text into sentences for optimal generation
        if SPACY_AVAILABLE:
            sentences = split_into_sentences(
                text, lang=language_id.lower() if language_id else "en"
            )
        else:
            sentences = [text]

        total_words = len(text.split())
        num_chunks = len(sentences)
        lang_name = (
            SUPPORTED_LANGUAGES.get(language_id.lower(), language_id)
            if language_id
            else "Unknown"
        )

        # Print generation plan
        print_generation_plan(
            total_words,
            sentences,
            "per-sentence",
            prefix=f"[Multilingual - {lang_name}] ",
        )

        # Generate audio for each sentence
        if len(sentences) == 1:
            # Single sentence - generate directly
            print_chunk_generating(0, 1, sentences[0])
            gen_start = _time.time()

            result = self._generate_single(
                sentences[0],
                language_id=language_id,
                cfg_weight=cfg_weight,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                show_progress=show_progress,  # Use caller's preference
            )

            gen_time = _time.time() - gen_start
            audio_duration = (
                result.shape[-1] / self.sr
                if hasattr(result, "shape")
                else len(result) / self.sr
            )
            print_chunk_completed(0, 1, gen_time, audio_duration)
            print_generation_complete(
                gen_time, audio_duration, 1, prefix="[Multilingual] "
            )

            # Force memory cleanup after generation
            clear_device_memory()

            return result

        # Multiple sentences - generate each and crossfade
        audio_chunks = []
        total_start = _time.time()

        for i, sentence in enumerate(sentences):
            len(sentence.split())
            print_chunk_generating(i, num_chunks, sentence)
            chunk_start = _time.time()

            chunk_audio = self._generate_single(
                sentence,
                language_id=language_id,
                cfg_weight=cfg_weight,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                show_progress=True,  # Show tqdm progress for each chunk
            )

            chunk_time = _time.time() - chunk_start
            chunk_duration = (
                chunk_audio.shape[-1] / self.sr
                if hasattr(chunk_audio, "shape")
                else len(chunk_audio) / self.sr
            )
            print_chunk_completed(i, num_chunks, chunk_time, chunk_duration)

            audio_chunks.append(chunk_audio)

        total_time = _time.time() - total_start

        # Crossfade chunks together
        print_crossfading(num_chunks)
        result = crossfade_chunks(audio_chunks, self.sr, 0.05)

        # Final summary
        result_np = result.numpy() if isinstance(result, torch.Tensor) else result
        total_audio_duration = len(result_np) / self.sr
        print_generation_complete(
            total_time, total_audio_duration, num_chunks, prefix="[Multilingual] "
        )

        # Force memory cleanup after generation
        clear_device_memory()

        return (
            torch.from_numpy(result_np).unsqueeze(0)
            if isinstance(result_np, np.ndarray)
            else result.unsqueeze(0)
        )

    def _generate_single(
        self,
        text: str,
        language_id: str,
        cfg_weight: float = 0.5,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.8,
        repetition_penalty: float = 2.0,
        min_p: float = 0.05,
        top_p: float = 1.0,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate speech for a single sentence/chunk (internal method).

        Args:
            text: Input text (single sentence/chunk)
            language_id: Language code
            cfg_weight: Classifier-free guidance weight
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            repetition_penalty: Repetition penalty factor
            min_p: Minimum probability threshold
            top_p: Nucleus sampling threshold
            show_progress: Whether to show token-level progress bar

        Returns:
            Generated audio waveform as torch tensor
        """
        # Normalize and tokenize
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(
            text, language_id=language_id.lower() if language_id else None
        ).to(self.device)

        # Estimate max_new_tokens if not provided
        if max_new_tokens is None:
            max_new_tokens = estimate_max_tokens(text, self.t3.hp.max_speech_tokens)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            # Determine device type and autocast settings
            if torch.cuda.is_available():
                device_type = "cuda"
                autocast_enabled = True
            else:
                device_type = "cpu"
                autocast_enabled = False

            with torch.autocast(
                device_type=device_type, dtype=torch.float16, enabled=autocast_enabled
            ):
                speech_tokens = self.t3.inference(
                    t3_cond=self.conds.t3,
                    text_tokens=text_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    repetition_penalty=repetition_penalty,
                    min_p=min_p,
                    top_p=top_p,
                    show_progress=show_progress,
                )
                speech_tokens = speech_tokens[0]
                speech_tokens = drop_invalid_tokens(speech_tokens)
                speech_tokens = speech_tokens.to(self.device)

                wav, sources = self.s3gen.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=self.conds.gen,
                )
                if sources is not None:
                    del sources
                wav = wav.squeeze(0).detach().cpu().numpy()

        return torch.from_numpy(wav).unsqueeze(0)

    def _split_text_intelligently(self, text, language_id, target_words_per_chunk=50):
        """
        Split text at sentence/phrase boundaries for chunked generation.
        Uses shared utility with spacy support.
        """
        return split_text_intelligently(
            text,
            target_words_per_chunk,
            lang=language_id.lower() if language_id else "en",
        )

    def _save_last_n_seconds(self, audio_tensor, output_path, duration=3.0):
        """
        Save the last N seconds of audio to a file for use as conditioning.

        Args:
            audio_tensor: Audio tensor (can be 1D or 2D)
            output_path: Path to save the audio
            duration: Duration in seconds to extract from the end
        """
        # Convert to numpy if needed
        if isinstance(audio_tensor, torch.Tensor):
            audio_np = audio_tensor.detach().cpu().numpy()
        else:
            audio_np = audio_tensor

        # Handle different shapes
        if audio_np.ndim == 2:
            audio_np = audio_np.squeeze(0)

        # Extract last N seconds
        num_samples = int(duration * self.sr)
        if len(audio_np) > num_samples:
            audio_np = audio_np[-num_samples:]

        # Save using librosa
        import soundfile as sf

        sf.write(output_path, audio_np, self.sr)

    def _crossfade_chunks(self, chunks, overlap_duration=0.1):
        """
        Concatenate audio chunks with crossfading.
        Uses shared optimized utility.
        """
        result = crossfade_chunks(chunks, self.sr, overlap_duration)
        return result.numpy() if isinstance(result, torch.Tensor) else result

    def generate_long(
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
        overlap_duration=0.05,
        max_new_tokens: Optional[int] = None,
        progress_callback=None,
        show_progress: bool = False,
    ):
        """
        Generate long-form speech with adaptive chunking strategy.

        Automatically chooses the best chunking strategy based on text length:
        - Short texts (< 50 words): Individual sentence processing
        - Long texts (>= 50 words): Grouped sentence processing (reduces overhead)

        Args:
            text: Input text to synthesize (any length)
            language_id: Language code (e.g., 'en', 'es', 'ja', 'zh')
            audio_prompt_path: Path to reference audio file for voice cloning
            exaggeration: Voice exaggeration/expressiveness level (0.0-1.0)
            cfg_weight: Classifier-free guidance weight (0.0-1.0)
            temperature: Sampling temperature for token generation
            repetition_penalty: Penalty for repeating tokens
            min_p: Minimum probability threshold for sampling
            top_p: Nucleus sampling threshold
            overlap_duration: Duration in seconds of crossfade between chunks
            max_new_tokens: Maximum tokens to generate per chunk
            progress_callback: Optional callback function for progress monitoring

        Returns:
            torch.Tensor: Generated audio waveform with shape (1, num_samples)
        """
        import time as _time

        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        # Prepare initial conditioning
        if audio_prompt_path:
            if progress_callback:
                progress_callback(
                    stage="preparing_conditionals", audio_path=audio_prompt_path
                )
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert (
                self.conds is not None
            ), "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
            _cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Get adaptive chunks based on text length
        lang = language_id.lower() if language_id else "en"
        chunks_to_generate, chunking_strategy = get_adaptive_chunks(text, lang=lang)

        if not chunks_to_generate:
            chunks_to_generate = [text]

        total_words = len(text.split())
        num_chunks = len(chunks_to_generate)
        lang_name = SUPPORTED_LANGUAGES.get(lang, language_id)

        if progress_callback:
            progress_callback(
                stage="text_split",
                total_chunks=num_chunks,
                chunk_previews=[
                    (i + 1, len(chunk.split()), chunk[:50])
                    for i, chunk in enumerate(chunks_to_generate)
                ],
            )

        # Print generation plan
        print_generation_plan(
            total_words,
            chunks_to_generate,
            chunking_strategy,
            prefix=f"[Multilingual - {lang_name}] ",
            is_long_form=True,
        )

        # Generate each chunk with status updates
        audio_chunks = []
        total_start = _time.time()

        for i, chunk_text in enumerate(chunks_to_generate):
            chunk_words = len(chunk_text.split())

            if progress_callback:
                progress_callback(
                    stage="chunk_start",
                    chunk_index=i,
                    chunk_number=i + 1,
                    total_chunks=num_chunks,
                    text_preview=chunk_text[:50],
                    word_count=chunk_words,
                )

            print_chunk_generating(i, num_chunks, chunk_text)
            chunk_start = _time.time()

            # Generate chunk
            chunk_audio = self._generate_single(
                chunk_text,
                language_id=language_id,
                cfg_weight=cfg_weight,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                show_progress=True,  # Show tqdm progress for each chunk
            )

            chunk_time = _time.time() - chunk_start
            chunk_duration = (
                chunk_audio.shape[-1] / self.sr
                if hasattr(chunk_audio, "shape")
                else len(chunk_audio) / self.sr
            )
            print_chunk_completed(i, num_chunks, chunk_time, chunk_duration)

            # MEMORY OPTIMIZATION: Move to CPU immediately
            if isinstance(chunk_audio, torch.Tensor):
                chunk_audio = chunk_audio.detach().cpu()
            audio_chunks.append(chunk_audio)

            # Aggressive memory cleanup after each chunk
            clear_device_memory()

            if progress_callback:
                progress_callback(
                    stage="chunk_complete",
                    chunk_index=i,
                    chunk_number=i + 1,
                    total_chunks=num_chunks,
                    audio_shape=(
                        chunk_audio.shape
                        if hasattr(chunk_audio, "shape")
                        else (len(chunk_audio),)
                    ),
                )

        total_time = _time.time() - total_start

        # Crossfade and concatenate all chunks
        if progress_callback:
            progress_callback(
                stage="crossfading",
                total_chunks=len(audio_chunks),
                overlap_duration=overlap_duration,
            )

        print_crossfading(num_chunks)
        result = crossfade_chunks(audio_chunks, self.sr, overlap_duration)

        # Final summary
        result_np = result.numpy() if isinstance(result, torch.Tensor) else result
        total_audio_duration = len(result_np) / self.sr
        print_generation_complete(
            total_time, total_audio_duration, num_chunks, prefix="[Multilingual] "
        )

        if progress_callback:
            progress_callback(
                stage="complete",
                total_chunks=len(audio_chunks),
                final_audio_shape=(1, len(result_np)),
            )

        return (
            torch.from_numpy(result_np).unsqueeze(0)
            if isinstance(result_np, np.ndarray)
            else result.unsqueeze(0)
        )
