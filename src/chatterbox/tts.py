from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
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
        ("“", '"'),
        ("”", '"'),
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
        arg_dict = dict(t3=self.t3.__dict__, gen=self.gen)
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs["t3"]), kwargs["gen"])


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
    def from_local(cls, ckpt_dir, device, t3_config=None) -> "ChatterboxTTS":
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device("cpu")
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve.to(device).eval()

        t3 = T3(hp=t3_config)
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(str(ckpt_dir / "tokenizer.json"))

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(
                device
            )

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device, t3_config=None) -> "ChatterboxTTS":
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print(
                    "MPS not available because the current PyTorch install was not built with MPS enabled."
                )
            else:
                print(
                    "MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine."
                )
            device = "cpu"

        for fpath in [
            "ve.safetensors",
            "t3_cfg.safetensors",
            "s3gen.safetensors",
            "tokenizer.json",
            "conds.pt",
        ]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device, t3_config=t3_config)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[: self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(
            s3gen_ref_wav, S3GEN_SR, device=self.device
        )

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward(
                [ref_16k_wav[: self.ENC_COND_LEN]], max_len=plen
            )
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(
                self.device
            )

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
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        max_new_tokens: Optional[int] = None,
        use_sentence_chunking: bool = True,
        overlap_duration: float = 0.05,
        language: str = "en",
        show_progress: bool = True,
    ):
        """
        Generate speech from text.

        By default, splits text into sentences for optimal generation quality.

        Args:
            text: Input text to synthesize
            repetition_penalty: Penalty for repeating tokens (1.0-3.0)
            min_p: Minimum probability threshold for sampling
            top_p: Nucleus sampling threshold
            audio_prompt_path: Optional path to reference audio for voice cloning
            exaggeration: Emotion exaggeration factor (0.0 to 1.0)
            cfg_weight: Classifier-free guidance weight
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate (auto-estimated if None)
            use_sentence_chunking: Whether to split text into sentences (default True)
            overlap_duration: Crossfade duration between sentences in seconds
            language: Language code for sentence tokenization (e.g., "en", "de", "fr")
            show_progress: Whether to show token-level progress bar (default True)

        Returns:
            Generated audio waveform as torch tensor
        """
        import time as _time

        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert (
                self.conds is not None
            ), "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm text
        text = punc_norm(text)

        # Split text into sentences for optimal generation
        if use_sentence_chunking and SPACY_AVAILABLE:
            sentences = split_into_sentences(text, lang=language)
        else:
            sentences = [text]

        total_words = len(text.split())
        num_chunks = len(sentences)

        # Print generation plan
        print_generation_plan(total_words, sentences, "per-sentence", prefix="")

        # Generate audio for each sentence
        if len(sentences) == 1:
            # Single sentence - generate directly with watermark
            print_chunk_generating(0, 1, sentences[0])
            gen_start = _time.time()

            result = self._generate_single(
                sentences[0],
                cfg_weight=cfg_weight,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                apply_watermark=True,
                show_progress=show_progress,  # Use caller's preference
            )

            gen_time = _time.time() - gen_start
            audio_duration = (
                result.shape[-1] / self.sr
                if hasattr(result, "shape")
                else len(result) / self.sr
            )
            print_chunk_completed(0, 1, gen_time, audio_duration)
            print_generation_complete(gen_time, audio_duration, 1)

            return result

        # Multiple sentences - generate each and crossfade
        audio_chunks = []
        total_start = _time.time()

        for i, sentence in enumerate(sentences):
            len(sentence.split())
            print_chunk_generating(i, num_chunks, sentence)
            chunk_start = _time.time()

            # Don't apply watermark to intermediate chunks
            chunk_audio = self._generate_single(
                sentence,
                cfg_weight=cfg_weight,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                apply_watermark=False,
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
        result = crossfade_chunks(audio_chunks, self.sr, overlap_duration)

        # Apply watermark to final concatenated audio
        result_np = result.numpy() if isinstance(result, torch.Tensor) else result
        watermarked_result = self.watermarker.apply_watermark(
            result_np, sample_rate=self.sr
        )

        # Final summary
        total_audio_duration = len(watermarked_result) / self.sr
        print_generation_complete(total_time, total_audio_duration, num_chunks)

        return torch.from_numpy(watermarked_result).unsqueeze(0)

    def _generate_single(
        self,
        text: str,
        cfg_weight: float = 0.5,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.8,
        top_p: float = 1.0,
        min_p: float = 0.05,
        repetition_penalty: float = 1.2,
        apply_watermark: bool = True,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate speech for a single sentence/chunk (internal method).

        Args:
            text: Input text (single sentence/chunk)
            cfg_weight: Classifier-free guidance weight
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            min_p: Minimum probability threshold
            repetition_penalty: Repetition penalty factor
            apply_watermark: Whether to apply watermark
            show_progress: Whether to show token-level progress bar

        Returns:
            Generated audio waveform as torch tensor
        """
        # Normalize and tokenize
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

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
            speech_tokens = speech_tokens[speech_tokens < 6561]
            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()

            if apply_watermark:
                wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)

        return torch.from_numpy(wav).unsqueeze(0)

    def _split_text_intelligently(self, text, target_words_per_chunk=50, language="en"):
        """
        Split text at sentence/phrase boundaries for chunked generation.
        Uses shared utility with spacy support.
        """
        return split_text_intelligently(text, target_words_per_chunk, lang=language)

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
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        overlap_duration=0.05,
        language: str = "en",
        max_new_tokens: Optional[int] = None,
        progress_callback=None,
        show_progress: bool = True,
    ):
        """
        Generate long-form speech with adaptive chunking strategy.

        Automatically chooses the best chunking strategy based on text length:
        - Short texts (< 50 words): Individual sentence processing
        - Long texts (>= 50 words): Grouped sentence processing (reduces overhead)

        Args:
            text: Input text to synthesize (any length)
            audio_prompt_path: Path to reference audio file for voice cloning
            exaggeration: Voice exaggeration/expressiveness level (0.0-1.0)
            cfg_weight: Classifier-free guidance weight (0.0-1.0)
            temperature: Sampling temperature for token generation
            repetition_penalty: Penalty for repeating tokens
            min_p: Minimum probability threshold for sampling
            top_p: Nucleus sampling threshold
            overlap_duration: Duration in seconds of crossfade between chunks
            language: Language code for sentence tokenization (e.g., "en", "de", "fr")
            max_new_tokens: Maximum tokens to generate per chunk
            progress_callback: Optional callback function for progress monitoring
            show_progress: Whether to show token-level progress bar (default True)

        Returns:
            torch.Tensor: Generated audio waveform with shape (1, num_samples)
        """
        import time as _time

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
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Get adaptive chunks based on text length
        chunks_to_generate, chunking_strategy = get_adaptive_chunks(text, lang=language)

        if not chunks_to_generate:
            chunks_to_generate = [text]

        total_words = len(text.split())
        num_chunks = len(chunks_to_generate)

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
            total_words, chunks_to_generate, chunking_strategy, is_long_form=True
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

            # Generate without watermark for intermediate chunks
            chunk_audio = self._generate_single(
                chunk_text,
                cfg_weight=cfg_weight,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                apply_watermark=False,
                show_progress=show_progress,  # Use caller's preference
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

        # Apply watermark to final concatenated audio
        result_np = result.numpy() if isinstance(result, torch.Tensor) else result
        watermarked_result = self.watermarker.apply_watermark(
            result_np, sample_rate=self.sr
        )

        # Final summary
        total_audio_duration = len(watermarked_result) / self.sr
        print_generation_complete(total_time, total_audio_duration, num_chunks)

        if progress_callback:
            progress_callback(
                stage="complete",
                total_chunks=len(audio_chunks),
                final_audio_shape=(1, len(watermarked_result)),
            )

        return torch.from_numpy(watermarked_result).unsqueeze(0)
