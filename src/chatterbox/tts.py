from dataclasses import dataclass
from pathlib import Path

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
                max_new_tokens=self.t3.hp.max_speech_tokens,
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

    def _split_text_intelligently(self, text, target_words_per_chunk=50):
        """
        Split text at sentence/phrase boundaries for chunked generation.

        Args:
            text: Input text to split
            target_words_per_chunk: Target number of words per chunk

        Returns:
            List of text chunks
        """
        import re

        # Split AFTER punctuation using lookbehind
        sentence_pattern = r'(?<=[.!?])\s+'

        # Split into sentences
        sentences = [s.strip() for s in re.split(sentence_pattern, text) if s.strip()]

        # Group sentences into chunks of approximately target_words_per_chunk
        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            if not sentence.strip():
                continue
            word_count = len(sentence.split())

            if current_word_count + word_count > target_words_per_chunk and current_chunk:
                # Start new chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_word_count = word_count
            else:
                current_chunk.append(sentence)
                current_word_count += word_count

        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks if chunks else [text]

    def _crossfade_chunks(self, chunks, overlap_duration=1.0):
        """
        Concatenate audio chunks with crossfading in overlap regions.

        Args:
            chunks: List of audio tensors (1D numpy arrays or tensors)
            overlap_duration: Duration of crossfade in seconds

        Returns:
            Single concatenated audio tensor
        """
        import numpy as np

        if len(chunks) == 0:
            return torch.tensor([])
        if len(chunks) == 1:
            chunk = chunks[0]
            if isinstance(chunk, torch.Tensor):
                chunk_np = chunk.detach().cpu().numpy()
            else:
                chunk_np = chunk
            # Ensure 1D output (squeeze if needed)
            if chunk_np.ndim == 2:
                chunk_np = chunk_np.squeeze(0)
            return chunk_np

        overlap_samples = int(overlap_duration * self.sr)

        # Convert all chunks to numpy
        np_chunks = []
        for chunk in chunks:
            if isinstance(chunk, torch.Tensor):
                chunk_np = chunk.detach().cpu().numpy()
            else:
                chunk_np = chunk
            if chunk_np.ndim == 2:
                chunk_np = chunk_np.squeeze(0)
            np_chunks.append(chunk_np)

        # Start with first chunk
        result = np_chunks[0]

        # Crossfade and concatenate remaining chunks
        for i in range(1, len(np_chunks)):
            current_chunk = np_chunks[i]

            if len(result) < overlap_samples or len(current_chunk) < overlap_samples:
                # Not enough samples for crossfade, just concatenate
                result = np.concatenate([result, current_chunk])
            else:
                # Create crossfade
                fade_out = np.linspace(1.0, 0.0, overlap_samples)
                fade_in = np.linspace(0.0, 1.0, overlap_samples)

                # Apply crossfade to overlapping region
                overlap_result = result[-overlap_samples:] * fade_out
                overlap_current = current_chunk[:overlap_samples] * fade_in
                overlap_mixed = overlap_result + overlap_current

                # Concatenate: result (minus overlap) + mixed overlap + rest of current
                result = np.concatenate([
                    result[:-overlap_samples],
                    overlap_mixed,
                    current_chunk[overlap_samples:]
                ])

        return result

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
        chunk_size_words=50,
        overlap_duration=0.1,
        progress_callback=None,
    ):
        """
        Generate long audio by chunking text with crossfading between chunks.

        This method is optimized for generating speech from long text (>50 words) without
        running out of memory or hitting generation limits. It splits the text into smaller
        chunks, generates audio for each chunk independently, and seamlessly combines them
        with crossfading.

        **When to use:**
        - Text longer than ~50 words
        - Documents, articles, or long-form content
        - When standard generate() produces truncated or poor-quality audio

        **How it works:**
        1. Splits text intelligently at sentence/phrase boundaries
        2. Generates audio for each chunk using the reference voice
        3. Crossfades chunks together for smooth transitions

        Args:
            text: Input text to synthesize (any length)
            audio_prompt_path: Path to reference audio file for voice cloning. If None, uses
                             previously prepared conditionals via `prepare_conditionals()`.
            exaggeration: Voice exaggeration/expressiveness level (0.0-1.0). Higher values
                        produce more expressive speech. Default: 0.5
            cfg_weight: Classifier-free guidance weight (0.0-1.0). Higher values follow
                      conditioning more closely. Default: 0.5
            temperature: Sampling temperature for token generation (0.0-2.0). Higher values
                       increase randomness. Default: 0.8
            repetition_penalty: Penalty for repeating tokens (1.0-3.0). Higher values reduce
                              repetition. Default: 1.2
            min_p: Minimum probability threshold for sampling. Default: 0.05
            top_p: Nucleus sampling threshold. Default: 1.0 (no filtering)
            chunk_size_words: Target number of words per chunk. Default: 50. Increase for
                            faster generation but higher memory usage; decrease for lower
                            memory usage.
            overlap_duration: Duration in seconds of crossfade between chunks. Default: 0.1.
                            For speech, shorter crossfades (0.05-0.2s) work better to avoid
                            blending words. Increase only for music-like content.
            progress_callback: Optional callback function to monitor generation progress.
                             Called with (stage, **kwargs) at various stages:
                             - "preparing_conditionals": Before loading reference audio
                             - "text_split": After splitting text into chunks
                             - "chunk_start": Before generating each chunk
                             - "chunk_complete": After generating each chunk
                             - "crossfading": Before combining chunks
                             - "complete": After final audio is ready

        Returns:
            torch.Tensor: Generated audio waveform with shape (1, num_samples) at 24kHz sample rate.

        Raises:
            AssertionError: If audio_prompt_path is None and conditionals haven't been prepared.

        Example:
            >>> model = ChatterboxTTS.from_pretrained(device="mps")
            >>> long_text = "This is a very long text that spans multiple sentences..."
            >>>
            >>> def progress_handler(stage, **kwargs):
            ...     if stage == "chunk_start":
            ...         print(f"Generating chunk {kwargs['chunk_number']}/{kwargs['total_chunks']}")
            >>>
            >>> wav = model.generate_long(
            ...     long_text,
            ...     audio_prompt_path="reference.wav",
            ...     chunk_size_words=50,
            ...     overlap_duration=1.0,
            ...     progress_callback=progress_handler
            ... )
            >>> import torchaudio as ta
            >>> ta.save("long_output.wav", wav, model.sr)
        """
        # Prepare initial conditioning
        if audio_prompt_path:
            if progress_callback:
                progress_callback(stage="preparing_conditionals", audio_path=audio_prompt_path)
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Split text into chunks
        text_chunks = self._split_text_intelligently(text, target_words_per_chunk=chunk_size_words)

        if progress_callback:
            progress_callback(
                stage="text_split",
                total_chunks=len(text_chunks),
                chunk_previews=[(i+1, len(chunk.split()), chunk[:50]) for i, chunk in enumerate(text_chunks)]
            )

        all_audio_chunks = []

        # Generate each chunk
        for i, text_chunk in enumerate(text_chunks):
            if progress_callback:
                progress_callback(
                    stage="chunk_start",
                    chunk_index=i,
                    chunk_number=i+1,
                    total_chunks=len(text_chunks),
                    text_preview=text_chunk[:50],
                    word_count=len(text_chunk.split())
                )

            # Generate audio for this chunk
            audio = self.generate(
                text_chunk,
                audio_prompt_path=None,  # Use cached conditionals
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )

            all_audio_chunks.append(audio)

            if progress_callback:
                progress_callback(
                    stage="chunk_complete",
                    chunk_index=i,
                    chunk_number=i+1,
                    total_chunks=len(text_chunks),
                    audio_shape=audio.shape
                )

        # Crossfade and concatenate all chunks
        if progress_callback:
            progress_callback(
                stage="crossfading",
                total_chunks=len(all_audio_chunks),
                overlap_duration=overlap_duration
            )

        final_audio = self._crossfade_chunks(all_audio_chunks, overlap_duration)

        if progress_callback:
            progress_callback(
                stage="complete",
                total_chunks=len(all_audio_chunks),
                final_audio_shape=(1, len(final_audio))
            )

        return torch.from_numpy(final_audio).unsqueeze(0)