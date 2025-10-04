from dataclasses import dataclass
from pathlib import Path
import os
import re
import tempfile
import numpy as np

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

        before_ve = get_memory_mb()
        print(f"before_ve: {before_ve:.1f}")
        ve = VoiceEncoder()
        ve.load_state_dict(
            torch.load(ckpt_dir / "ve.pt", weights_only=True)
        )
        ve.to(device).eval()

        before_t3 = get_memory_mb()
        print(f"before_t3: {before_t3:.1f}")
        t3 = T3(T3Config.multilingual())
        t3_state = load_safetensors(ckpt_dir / "t3_23lang.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        before_s3_gen = get_memory_mb()
        print(f"before_s3_gen: {before_s3_gen:.1f}")
        s3gen = S3Gen()
        s3gen.load_state_dict(
            torch.load(ckpt_dir / "s3gen.pt", weights_only=True)
        )
        s3gen.to(device).eval()
        before_tokenizer = get_memory_mb()
        print(f"before_tokenizer: {before_tokenizer:.1f}")
        tokenizer = MTLTokenizer(
            str(ckpt_dir / "mtl_tokenizer.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice).to(device)
        after_from_local_load = get_memory_mb()
        print(f"after_from_local_load: {after_from_local_load:.1f}")
        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device: torch.device) -> 'ChatterboxMultilingualTTS':
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
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            # Limit audio length more aggressively for memory safety
            # Increased from 3 to 6 seconds for better conditioning
            safe_audio_len = min(len(ref_16k_wav), 6 * S3_SR)
            limited_audio = ref_16k_wav[:safe_audio_len]

            # Add memory cleanup before tokenization
            import gc
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

            # Use smaller max_len to be extra safe
            safe_max_len = min(plen, 150)  # 150 tokens for better quality
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([limited_audio], max_len=safe_max_len)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

            # More memory cleanup after tokenization
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

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
        language_id,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=2.0,
        min_p=0.05,
        top_p=1.0,
    ):
        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )
        
        if audio_prompt_path:
            before_audio_prompt = get_memory_mb()
            print(f"before_audio_prompt: {before_audio_prompt:.1f}")
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
            after_audio_prompt = get_memory_mb()
            print(f"after_audio_prompt: {after_audio_prompt:.1f}")
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed

        before_exaggeration = get_memory_mb()
        print(f"before_exaggeration: {before_exaggeration:.1f}")
        if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        after_exaggeration = get_memory_mb()
        print(f"after_exaggeration: {after_exaggeration:.1f}")

        # Norm and tokenize text
        text = punc_norm(text)

        after_tokenize_text = get_memory_mb()
        print(f"after_tokenize_text: {after_tokenize_text:.1f}")
        text_tokens = self.tokenizer.text_to_tokens(text, language_id=language_id.lower() if language_id else None).to(self.device)
        after_text_to_tokens = get_memory_mb()
        print(f"after_text_to_tokens: {after_text_to_tokens:.1f}")
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        after_torch_cat = get_memory_mb()
        print(f"after_torch_cat: {after_torch_cat:.1f}")

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token

        after_eot = get_memory_mb()
        print(f"after_eot: {after_eot:.1f}")
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        generate_baseline = get_memory_mb()
        print(f"generate_baseline: {generate_baseline:.1f}")
        with torch.inference_mode():
            # Use mixed precision inference for better memory efficiency
            with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16, enabled=torch.cuda.is_available()):
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
                after_speech_tokens_created = get_memory_mb()
                print(f"after_speech_tokens_created: {after_speech_tokens_created:.1f}")
                # Extract only the conditional batch.
                speech_tokens = speech_tokens[0]

                # TODO: output becomes 1D
                speech_tokens = drop_invalid_tokens(speech_tokens)
                speech_tokens = speech_tokens.to(self.device)

                del text_tokens

                before_s3gen_inference = get_memory_mb()
                print(f"before_s3gen_inference: {before_s3gen_inference:.1f}")
                wav, sources = self.s3gen.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=self.conds.gen,
                )
                if sources is not None:
                    del sources
                after_s3gen_inference = get_memory_mb()

                del speech_tokens
                wav = wav.squeeze(0).detach().cpu().numpy()

                print(f"after_s3gen_inference: {after_s3gen_inference:.1f}")
        return torch.from_numpy(wav).unsqueeze(0)

    def _split_text_intelligently(self, text, language_id, target_words_per_chunk=50):
        """
        Split text at sentence/phrase boundaries for chunked generation.

        Args:
            text: Input text to split
            language_id: Language code for language-specific splitting
            target_words_per_chunk: Target number of words per chunk

        Returns:
            List of text chunks
        """
        # Define sentence endings based on language
        if language_id in ['ja', 'zh']:
            # Japanese and Chinese sentence endings
            sentence_pattern = r'[。！？\.!?]+'
        else:
            # Most other languages
            sentence_pattern = r'[.!?]+\s+'

        # Split into sentences
        sentences = re.split(f'({sentence_pattern})', text)
        # Recombine sentences with their punctuation
        sentences = [''.join(sentences[i:i+2]).strip() for i in range(0, len(sentences)-1, 2)]
        if len(sentences) * 2 < len(text.split(sentence_pattern)):
            # Add last sentence if it exists
            last = text.split(sentences[-1])[-1].strip() if sentences else text
            if last:
                sentences.append(last)

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

    def _crossfade_chunks(self, chunks, overlap_duration=1.0):
        """
        Concatenate audio chunks with crossfading in overlap regions.

        Args:
            chunks: List of audio tensors (1D numpy arrays or tensors)
            overlap_duration: Duration of crossfade in seconds

        Returns:
            Single concatenated audio tensor
        """
        if len(chunks) == 0:
            return torch.tensor([])
        if len(chunks) == 1:
            chunk = chunks[0]
            if isinstance(chunk, torch.Tensor):
                return chunk.detach().cpu().numpy()
            return chunk

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
        language_id,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=2.0,
        min_p=0.05,
        top_p=1.0,
        chunk_size_words=50,
        overlap_duration=1.0,
    ):
        """
        Generate long audio by chunking text and using sliding window conditioning.

        This method splits long text into smaller chunks, generates audio for each chunk,
        and uses the end of each generated chunk as conditioning for the next one.
        This maintains voice consistency without memory issues.

        Args:
            text: Input text to synthesize
            language_id: Language code (e.g., 'en', 'es', 'ja')
            audio_prompt_path: Path to reference audio for voice cloning
            exaggeration: Voice exaggeration level (0.0-1.0)
            cfg_weight: Classifier-free guidance weight
            temperature: Sampling temperature
            repetition_penalty: Penalty for token repetition
            min_p: Minimum probability threshold
            top_p: Top-p sampling threshold
            chunk_size_words: Target words per chunk (default: 50)
            overlap_duration: Crossfade duration in seconds (default: 1.0)

        Returns:
            Audio tensor with shape (1, samples)
        """
        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        # Prepare initial conditioning
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Split text into chunks
        text_chunks = self._split_text_intelligently(text, language_id, target_words_per_chunk=chunk_size_words)

        print(f"Split text into {len(text_chunks)} chunks")
        for i, chunk in enumerate(text_chunks):
            print(f"  Chunk {i+1}: {len(chunk.split())} words - '{chunk[:50]}...'")

        all_audio_chunks = []
        current_conditioning = audio_prompt_path

        # Generate each chunk
        for i, text_chunk in enumerate(text_chunks):
            print(f"\nGenerating chunk {i+1}/{len(text_chunks)}...")

            # Generate audio for this chunk
            audio = self.generate(
                text_chunk,
                language_id,
                audio_prompt_path=current_conditioning,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )

            all_audio_chunks.append(audio)

            # Save last N seconds as conditioning for next chunk
            if i < len(text_chunks) - 1:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    temp_path = tmp_file.name
                    self._save_last_n_seconds(audio, temp_path, duration=3.0)
                    current_conditioning = temp_path

        # Crossfade and concatenate all chunks
        print(f"\nCrossfading {len(all_audio_chunks)} chunks...")
        final_audio = self._crossfade_chunks(all_audio_chunks, overlap_duration)

        # Clean up temporary files
        if current_conditioning != audio_prompt_path and os.path.exists(current_conditioning):
            try:
                os.unlink(current_conditioning)
            except:
                pass

        return torch.from_numpy(final_audio).unsqueeze(0)
