import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset as hf_dataset
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder
from transformers import GPT2TokenizerFast

from chatterbox.models.s3tokenizer import S3_SR, S3Tokenizer
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.tokenizers.tokenizer import EnTokenizer, MTLTokenizer
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts import ChatterboxTTS, punc_norm
from chatterbox.tts_turbo import ChatterboxTurboTTS
from finetune.custom_arguments import DataArguments

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {
    "da": "Danish",
    "en": "English",
}


# --- Dataset Class ---
class SpeechFineTuningDataset(Dataset):
    def __init__(
        self,
        data_args: DataArguments,
        chatterbox_model: ChatterboxTTS | ChatterboxMultilingualTTS | ChatterboxTurboTTS,
        t3_config: T3Config,
        hf_dataset: Union[hf_dataset, List[Dict[str, str]]],
    ):
        self.data_args = data_args
        self.t3_config = t3_config
        self.dataset_source = hf_dataset

        self.is_multilingual = isinstance(chatterbox_model, ChatterboxMultilingualTTS)
        self.text_tokenizer: GPT2TokenizerFast | EnTokenizer | MTLTokenizer = chatterbox_model.tokenizer
        self.speech_tokenizer: S3Tokenizer = chatterbox_model.s3gen.tokenizer
        self.voice_encoder = chatterbox_model.ve

        self.enc_cond_audio_len_samples = int(data_args.audio_prompt_duration_s * S3_SR)

    def __len__(self) -> int:
        return len(self.dataset_source)

    # --- Audio loading ---

    def _load_and_resample_audio(self, item: Dict, idx: int) -> Optional[np.ndarray]:
        """Load audio from the dataset item, resample to 16 kHz, and return as float32 mono."""
        audio_data = item[self.data_args.audio_column_name]

        if isinstance(audio_data, str):
            wav, original_sr = librosa.load(audio_data, sr=None, mono=True)
        elif isinstance(audio_data, dict) and "array" in audio_data and "sampling_rate" in audio_data:
            wav = audio_data["array"]
            original_sr = audio_data["sampling_rate"]
        elif isinstance(audio_data, AudioDecoder):
            wav = audio_data["array"]  # type: ignore[index]
            sr = audio_data._desired_sample_rate  # type: ignore[attr-defined]
            if sr is None:
                logger.error(f"Item {idx}: AudioDecoder has no sample rate. Skipping.")
                return None
            original_sr = float(sr)
        else:
            logger.error(f"Item {idx}: unexpected audio format {type(audio_data)}. Skipping.")
            return None

        if not isinstance(wav, np.ndarray):
            logger.error(f"Item {idx}: audio array is not numpy ({type(wav)}). Skipping.")
            return None

        if original_sr != S3_SR:
            wav = librosa.resample(wav, orig_sr=float(original_sr), target_sr=S3_SR)
        if wav.ndim > 1:
            wav = librosa.to_mono(wav)
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)

        return wav

    # --- Text tokenization ---

    def _tokenize_text(self, text: str, language_id: Optional[str], idx: int) -> Optional[torch.Tensor]:
        """Tokenize text, prepend SOS, append EOS, and truncate to max_text_len.

        Returns a 1-D long tensor on CPU, or None if the item should be skipped.
        """
        normalized = punc_norm(text)
        try:
            if isinstance(self.text_tokenizer, GPT2TokenizerFast):
                # Turbo model: normalization not applied to match original tokenizer expectations
                raw_tokens = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                raw_tokens = raw_tokens.input_ids.squeeze(0)
            elif isinstance(self.text_tokenizer, MTLTokenizer):
                if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
                    logger.warning(f"Item {idx}: unsupported language '{language_id}'. Skipping.")
                    return None
                raw_tokens = self.text_tokenizer.text_to_tokens(
                    normalized, language_id=language_id.lower() if language_id else None  # type: ignore[arg-type]
                ).squeeze(0)
            else:
                raw_tokens = self.text_tokenizer.text_to_tokens(normalized).squeeze(0)
        except Exception as e:
            logger.error(f"Item {idx}: error tokenizing text: {e}. Skipping.")
            return None

        tokens = F.pad(raw_tokens, (1, 0), value=self.t3_config.start_text_token)
        tokens = F.pad(tokens, (0, 1), value=self.t3_config.stop_text_token)
        if len(tokens) > self.data_args.max_text_len:
            tokens = tokens[: self.data_args.max_text_len - 1]
            tokens = torch.cat([tokens, torch.tensor([self.t3_config.stop_text_token])])

        return tokens.long()

    # --- Speech tokenization ---

    def _tokenize_speech(self, wav_16k: np.ndarray, idx: int) -> Optional[torch.Tensor]:
        """Tokenize speech waveform, prepend SOS, append EOS, and truncate to max_speech_len.

        Returns a 1-D long tensor on CPU, or None if the item should be skipped.
        """
        try:
            tokens_batch, lens_batch = self.speech_tokenizer.forward([wav_16k])  # type: ignore[arg-type]
            if tokens_batch is None or lens_batch is None:
                logger.error(f"Item {idx}: S3Tokenizer returned None. Skipping.")
                return None
            tokens = tokens_batch.squeeze(0)[: int(lens_batch.squeeze(0).item())].cpu()
        except Exception as e:
            logger.error(f"Item {idx}: error tokenizing speech: {e}. Skipping.")
            return None

        tokens = F.pad(tokens, (1, 0), value=self.t3_config.start_speech_token)
        tokens = F.pad(tokens, (0, 1), value=self.t3_config.stop_speech_token)
        if len(tokens) > self.data_args.max_speech_len:
            tokens = tokens[: self.data_args.max_speech_len - 1]
            tokens = torch.cat([tokens, torch.tensor([self.t3_config.stop_speech_token])])

        return tokens.long()

    # --- Speaker embedding ---

    def _get_speaker_embedding(self, item: Dict, wav_16k: np.ndarray, idx: int) -> Optional[torch.Tensor]:
        """Return speaker embedding as a CPU float tensor.

        Uses a precomputed embedding from the dataset item when available,
        otherwise runs the voice encoder on the waveform.
        """
        precomputed = item.get("speaker_embedding", None)
        if precomputed is not None:
            return torch.tensor(precomputed, dtype=torch.float)

        try:
            emb_np = self.voice_encoder.embeds_from_wavs([wav_16k], sample_rate=S3_SR)
            return torch.from_numpy(emb_np[0])
        except Exception as e:
            logger.error(f"Item {idx}: error computing speaker embedding: {e}. Skipping.")
            return None

    # --- Conditioning tokens ---

    def _compute_cond_tokens(self, wav_16k: np.ndarray) -> tuple[torch.Tensor, int]:
        """Extract conditioning speech tokens from the beginning of the waveform.

        Returns:
            tokens: 1-D long tensor of length speech_cond_prompt_len (CPU).
            actual_len: real token count before any padding (used for label masking).
        """
        target_len = self.t3_config.speech_cond_prompt_len
        zeros = torch.zeros(target_len, dtype=torch.long)

        cond_audio = wav_16k[: self.enc_cond_audio_len_samples]
        if len(cond_audio) == 0:
            return zeros, 0

        try:
            tokens_batch, token_lens = self.speech_tokenizer.forward(
                [cond_audio], max_len=target_len  # type: ignore[arg-type]
            )
            if tokens_batch is None:
                return zeros, 0
            tokens = tokens_batch.squeeze(0).cpu()
            actual_len: int = int(token_lens.item()) if token_lens is not None else tokens.size(0)
            return tokens, actual_len
        except Exception as e:
            logger.error(f"Error computing conditioning tokens: {e}. Returning zeros.")
            return zeros, 0

    def _get_cond_tokens(self, item: Dict, wav_16k: np.ndarray) -> tuple[torch.Tensor, int, bool]:
        """Return conditioning tokens, their actual (pre-padding) length, and a cond_from_ref flag.

        The cond_from_ref flag indicates whether the conditioning tokens came from a separate
        reference audio clip (True) or from the training sample itself (False). The data
        collator uses this flag to decide whether to mask prompt positions in labels_speech:
        if the model is conditioned on the same audio it is trying to predict, masking those
        positions prevents the model from trivially copying the conditioning input.

        Uses precomputed tokens from the dataset item when available (cond_from_ref=True),
        otherwise extracts tokens from the beginning of wav_16k (cond_from_ref=False).
        """
        precomputed = item.get("cond_prompt_speech_tokens", None)
        if precomputed is not None:
            tokens = torch.tensor(precomputed, dtype=torch.long)
            actual_len = len(precomputed)
            cond_from_ref = True
        else:
            tokens, actual_len = self._compute_cond_tokens(wav_16k)
            cond_from_ref = False

        # Pad or truncate to the config-specified length
        target_len = self.t3_config.speech_cond_prompt_len
        current_len = tokens.size(0)
        if current_len > target_len:
            tokens = tokens[:target_len]
            actual_len = min(actual_len, target_len)
        elif current_len < target_len:
            tokens = F.pad(tokens, (0, target_len - current_len), value=0)

        return tokens, actual_len, cond_from_ref

    # --- Main item loading ---

    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        item = self.dataset_source[idx]
        text = item[self.data_args.text_column_name]
        language_id = item.get("language") if self.is_multilingual else None

        wav_16k = self._load_and_resample_audio(item, idx)
        if wav_16k is None or len(wav_16k) == 0:
            return None

        text_tokens = self._tokenize_text(text, language_id, idx)
        if text_tokens is None:
            return None

        speech_tokens = self._tokenize_speech(wav_16k, idx)
        if speech_tokens is None:
            return None

        speaker_emb = self._get_speaker_embedding(item, wav_16k, idx)
        if speaker_emb is None:
            return None

        cond_tokens, actual_cond_len, cond_from_ref = self._get_cond_tokens(item, wav_16k)

        # No 'exaggeration' level available in the dataset; use the neutral midpoint
        result: Dict[str, Any] = {
            "text_tokens": text_tokens,
            "text_token_lens": torch.tensor(len(text_tokens), dtype=torch.long),
            "speech_tokens": speech_tokens,
            "speech_token_lens": torch.tensor(len(speech_tokens), dtype=torch.long),
            "t3_cond_speaker_emb": speaker_emb.float(),
            "t3_cond_prompt_speech_tokens": cond_tokens,
            "t3_cond_prompt_len": torch.tensor(actual_cond_len, dtype=torch.long),
            "t3_cond_emotion_adv": torch.tensor(0.5, dtype=torch.float),
            "cond_from_ref": torch.tensor(cond_from_ref, dtype=torch.bool),
        }

        for key, val in result.items():
            if isinstance(val, torch.Tensor) and (torch.isnan(val).any() or torch.isinf(val).any()):
                logger.error(f"Item {idx}: tensor '{key}' contains NaN or Inf. Skipping.")
                return None

        return result


# --- Data Collator ---
@dataclass
class SpeechDataCollator:
    text_pad_token_id: int
    speech_pad_token_id: int

    def __call__(self, features: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
        valid_features: List[Dict[str, Any]] = [f for f in features if f is not None]
        if not valid_features:
            logger.warning("SpeechDataCollator received no valid features. Returning empty batch.")
            return {}

        batch_size = len(valid_features)
        text_tokens_list = [f["text_tokens"] for f in valid_features]
        speech_tokens_list = [f["speech_tokens"] for f in valid_features]
        max_text_len = max(len(t) for t in text_tokens_list)
        max_speech_len = max(len(s) for s in speech_tokens_list)

        # Pad text and speech tokens
        padded_text_tokens = torch.stack(
            [F.pad(t, (0, max_text_len - len(t)), value=self.text_pad_token_id) for t in text_tokens_list]
        )  # (B, max_text_len)
        padded_speech_tokens = torch.stack(
            [F.pad(s, (0, max_speech_len - len(s)), value=self.speech_pad_token_id) for s in speech_tokens_list]
        )  # (B, max_speech_len)

        # Collect lengths
        text_token_lens = torch.stack([f["text_token_lens"] for f in valid_features])  # (B,)
        speech_token_lens = torch.stack([f["speech_token_lens"] for f in valid_features])  # (B,)

        # Collect conditionals
        t3_cond_speaker_emb = torch.stack([f["t3_cond_speaker_emb"] for f in valid_features])  # (B, D_speaker)
        t3_cond_prompt_speech_tokens = torch.stack([f["t3_cond_prompt_speech_tokens"] for f in valid_features])  # (B, prompt_len)
        t3_cond_emotion_adv = torch.stack([f["t3_cond_emotion_adv"] for f in valid_features]).view(batch_size, 1, 1)  # (B, 1, 1)

        IGNORE_ID = -100

        # --- Build labels_text ---
        # labels_text[b, t] = text_tokens[b, t+1], masked to IGNORE_ID at and beyond EOS
        shifted_text = padded_text_tokens[:, 1:].contiguous()  # (B, max_text_len - 1)
        T_text = shifted_text.size(1)
        text_lens_minus_one = (text_token_lens - 1).clamp(min=0)  # (B,)
        mask_pad_text = torch.arange(T_text)[None] >= text_lens_minus_one[:, None]  # (B, T_text)
        labels_text = shifted_text.masked_fill(mask_pad_text, IGNORE_ID)  # (B, T_text)

        # --- Build labels_speech ---
        # labels_speech[b, t] = speech_tokens[b, t+1], masked to IGNORE_ID at and beyond EOS
        shifted_speech = padded_speech_tokens[:, 1:].contiguous()  # (B, max_speech_len - 1)
        T_speech = shifted_speech.size(1)
        speech_lens_minus_one = (speech_token_lens - 1).clamp(min=0)  # (B,)
        mask_pad_speech = torch.arange(T_speech)[None] >= speech_lens_minus_one[:, None]  # (B, T_speech)

        # Mask prompt positions for samples where conditioning is from the same audio.
        # When cond_from_ref=False the first actual_cond_len tokens duplicate the conditioning
        # input — masking prevents the model from trivially copying them.
        # When cond_from_ref=True the conditioning came from a different clip, so nothing to mask.
        cond_from_ref_flags = torch.stack([f["cond_from_ref"] for f in valid_features])  # (B,)
        actual_cond_lens = torch.stack([f["t3_cond_prompt_len"] for f in valid_features])  # (B,)
        mask_prompt = torch.zeros(batch_size, T_speech, dtype=torch.bool)
        for i in range(batch_size):
            if not cond_from_ref_flags[i].item():
                mask_prompt[i, : int(actual_cond_lens[i].item())] = True

        labels_speech = shifted_speech.masked_fill(mask_pad_speech | mask_prompt, IGNORE_ID)  # (B, T_speech)

        batch_dict: Dict[str, Any] = {
            "text_tokens": padded_text_tokens,
            "text_token_lens": text_token_lens,
            "speech_tokens": padded_speech_tokens,
            "speech_token_lens": speech_token_lens,
            "t3_cond_speaker_emb": t3_cond_speaker_emb,
            "t3_cond_prompt_speech_tokens": t3_cond_prompt_speech_tokens,
            "t3_cond_emotion_adv": t3_cond_emotion_adv,
            "labels_text": labels_text,      # (B, max_text_len - 1), IGNORE_ID where masked
            "labels_speech": labels_speech,  # (B, max_speech_len - 1), IGNORE_ID where masked
        }

        return batch_dict
