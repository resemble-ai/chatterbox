from __future__ import annotations

import asyncio
import inspect
import os
import warnings
from pathlib import Path
from typing import AsyncGenerator, Generator, Literal

import torch
from huggingface_hub import snapshot_download

from .mtl_tts import ChatterboxMultilingualTTS
from .tts import ChatterboxTTS
from .tts_turbo import ChatterboxTurboTTS
from .utils.normalizer import normalize_text as normalize_text_content
from .utils.splitter import split_sentences


ModelType = ChatterboxTTS | ChatterboxMultilingualTTS | ChatterboxTurboTTS


class ChatterboxInference:
    """Thin inference wrapper around existing Chatterbox TTS models.

    Not thread-safe. Speaker conditioning state (conds, CUDA graph cache) is stored
    on the instance. Use one instance per process or worker; do not share across threads
    or concurrent requests.
    """

    MODEL_ALLOW_PATTERNS = {
        "base": ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"],
        "multilingual": [
            "ve.pt",
            "t3_mtl23ls_v2.safetensors",
            "s3gen.pt",
            "grapheme_mtl_merged_expanded_v1.json",
            "conds.pt",
            "Cangjie5_TC.json",
        ],
        "turbo": ["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
    }

    def __init__(
        self,
        model: ModelType,
        language: str = "en",
        normalize_text: bool = True,
        sentence_split: bool = True,
        inter_sentence_silence_ms: int = 100,
    ):
        self.model = model
        self.language = language
        self.normalize_text = normalize_text
        self.sentence_split = sentence_split
        self.inter_sentence_silence_ms = inter_sentence_silence_ms
        self.sr = getattr(model, "sr", 24000)
        self._last_audio_prompt_path: str | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_type: Literal["base", "multilingual", "turbo"] = "multilingual",
        language: str = "en",
        device=None,
        repo_id: str | None = None,
        normalize_text: bool = True,
        sentence_split: bool = True,
        inter_sentence_silence_ms: int = 100,
    ) -> "ChatterboxInference":
        if repo_id is not None:
            model = cls._load_model_from_repo(model_type=model_type, repo_id=repo_id, device=device)
        else:
            model = cls._load_default_model(model_type=model_type, device=device)

        return cls(
            model=model,
            language=language,
            normalize_text=normalize_text,
            sentence_split=sentence_split,
            inter_sentence_silence_ms=inter_sentence_silence_ms,
        )

    @classmethod
    def from_local(
        cls,
        ckpt_dir: str | Path,
        model_type: Literal["base", "multilingual", "turbo"] = "multilingual",
        language: str = "en",
        device=None,
        normalize_text: bool = True,
        sentence_split: bool = True,
        inter_sentence_silence_ms: int = 100,
    ) -> "ChatterboxInference":
        model = cls._load_model_from_local(model_type=model_type, ckpt_dir=ckpt_dir, device=device)

        return cls(
            model=model,
            language=language,
            normalize_text=normalize_text,
            sentence_split=sentence_split,
            inter_sentence_silence_ms=inter_sentence_silence_ms,
        )

    @classmethod
    def from_model(
        cls,
        model: ModelType,
        language: str = "en",
        normalize_text: bool = True,
        sentence_split: bool = True,
        inter_sentence_silence_ms: int = 100,
    ) -> "ChatterboxInference":
        return cls(
            model=model,
            language=language,
            normalize_text=normalize_text,
            sentence_split=sentence_split,
            inter_sentence_silence_ms=inter_sentence_silence_ms,
        )

    @classmethod
    def _load_default_model(cls, model_type: str, device=None) -> ModelType:
        if model_type == "multilingual":
            return ChatterboxMultilingualTTS.from_pretrained(device=device)
        if model_type == "turbo":
            return ChatterboxTurboTTS.from_pretrained(device=device)
        return ChatterboxTTS.from_pretrained(device=device)

    @classmethod
    def _load_model_from_local(cls, model_type: str, ckpt_dir: str | Path, device=None) -> ModelType:
        if model_type == "multilingual":
            return ChatterboxMultilingualTTS.from_local(ckpt_dir, device=device)
        if model_type == "turbo":
            return ChatterboxTurboTTS.from_local(ckpt_dir, device=device)
        return ChatterboxTTS.from_local(ckpt_dir, device=device)

    @classmethod
    def _load_model_from_repo(cls, model_type: str, repo_id: str, device=None) -> ModelType:
        ckpt_dir = snapshot_download(
            repo_id=repo_id,
            token=os.getenv("HF_TOKEN") or True,
            allow_patterns=cls.MODEL_ALLOW_PATTERNS[model_type],
        )
        return cls._load_model_from_local(model_type=model_type, ckpt_dir=ckpt_dir, device=device)

    def prepare_conditionals(self, audio_prompt_path: str, **kwargs) -> None:
        """Pre-compute and cache speaker embeddings from a reference audio file.

        Call this once before generate() to avoid re-encoding on every sentence.
        Accepts the same kwargs as the underlying model's prepare_conditionals()
        (e.g. exaggeration, norm_loudness for turbo).
        """
        valid_params = set(inspect.signature(self.model.prepare_conditionals).parameters) - {"self", "wav_fpath"}
        filtered = {k: v for k, v in kwargs.items() if k in valid_params}
        self.model.prepare_conditionals(audio_prompt_path, **filtered)
        self._last_audio_prompt_path = audio_prompt_path

    def _prepare_text(
        self,
        text: str,
        language_id: str | None = None,
        normalize_text: bool | None = None,
        sentence_split: bool | None = None,
    ) -> list[str]:
        language = language_id or self.language
        use_normalization = self.normalize_text if normalize_text is None else normalize_text
        use_sentence_split = self.sentence_split if sentence_split is None else sentence_split

        processed_text = text
        if use_normalization:
            processed_text = normalize_text_content(processed_text, language=language)

        processed_text = processed_text.strip()
        if not processed_text:
            return []

        if not use_sentence_split:
            return [processed_text]

        return split_sentences(processed_text, language=language)

    def _silence_chunk(self, inter_sentence_silence_ms: int | None = None) -> torch.Tensor | None:
        silence_ms = self.inter_sentence_silence_ms if inter_sentence_silence_ms is None else inter_sentence_silence_ms
        if silence_ms <= 0:
            return None

        samples = int(self.sr * silence_ms / 1000)
        if samples <= 0:
            return None

        return torch.zeros((1, samples), dtype=torch.float32)

    def generate(
        self,
        text: str,
        language_id: str | None = None,
        normalize_text: bool | None = None,
        sentence_split: bool | None = None,
        inter_sentence_silence_ms: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        sentences = self._prepare_text(
            text,
            language_id=language_id,
            normalize_text=normalize_text,
            sentence_split=sentence_split,
        )
        if not sentences:
            return torch.zeros((1, 0), dtype=torch.float32)

        chunks = []
        silence = self._silence_chunk(inter_sentence_silence_ms=inter_sentence_silence_ms)

        if isinstance(self.model, ChatterboxMultilingualTTS):
            kwargs["language_id"] = language_id or self.language

        # Pre-compute speaker embeddings once for the full generate call.
        # Only re-run if the audio prompt path has changed since the last call.
        audio_prompt_path = kwargs.pop("audio_prompt_path", None)
        if audio_prompt_path and audio_prompt_path != self._last_audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, **kwargs)

        valid_params = set(inspect.signature(self.model.generate).parameters)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        dropped = set(kwargs) - valid_params
        if dropped:
            warnings.warn(
                f"{type(self.model).__name__}.generate() does not accept: {sorted(dropped)}. These kwargs were ignored.",
                stacklevel=2,
            )

        for index, sentence in enumerate(sentences):
            chunks.append(self.model.generate(sentence, **filtered_kwargs))
            if silence is not None and index < len(sentences) - 1:
                chunks.append(silence)

        return torch.cat(chunks, dim=-1)

    def generate_fast(
        self,
        text: str,
        language_id: str | None = None,
        normalize_text: bool | None = None,
        sentence_split: bool | None = None,
        inter_sentence_silence_ms: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Fast inference using CUDA graphs. Falls back to generate() on non-CUDA devices.
        See generate() for parameter documentation."""
        if not hasattr(self.model, "generate_fast"):
            return self.generate(
                text,
                language_id=language_id,
                normalize_text=normalize_text,
                sentence_split=sentence_split,
                inter_sentence_silence_ms=inter_sentence_silence_ms,
                **kwargs,
            )

        sentences = self._prepare_text(
            text,
            language_id=language_id,
            normalize_text=normalize_text,
            sentence_split=sentence_split,
        )
        if not sentences:
            return torch.zeros((1, 0), dtype=torch.float32)

        chunks = []
        silence = self._silence_chunk(inter_sentence_silence_ms=inter_sentence_silence_ms)

        if isinstance(self.model, ChatterboxMultilingualTTS):
            kwargs["language_id"] = language_id or self.language

        audio_prompt_path = kwargs.pop("audio_prompt_path", None)
        if audio_prompt_path and audio_prompt_path != self._last_audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, **kwargs)

        valid_params = set(inspect.signature(self.model.generate_fast).parameters)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        dropped = set(kwargs) - valid_params
        if dropped:
            warnings.warn(
                f"{type(self.model).__name__}.generate_fast() does not accept: {sorted(dropped)}. These kwargs were ignored.",
                stacklevel=2,
            )

        for index, sentence in enumerate(sentences):
            chunks.append(self.model.generate_fast(sentence, **filtered_kwargs))
            if silence is not None and index < len(sentences) - 1:
                chunks.append(silence)

        return torch.cat(chunks, dim=-1)

    def generate_stream_sync(
        self,
        text: str,
        language_id: str | None = None,
        normalize_text: bool | None = None,
        inter_sentence_silence_ms: int | None = None,
        **kwargs,
    ) -> Generator[torch.Tensor, None, None]:
        """Sync generator yielding one wav tensor per sentence (plus silence tensors between).
        Sentence splitting is always enabled — each yielded chunk corresponds to one sentence.
        Concatenating all yielded tensors produces the same result as generate()."""
        sentences = self._prepare_text(
            text,
            language_id=language_id,
            normalize_text=normalize_text,
            sentence_split=True,
        )
        if not sentences:
            return

        silence = self._silence_chunk(inter_sentence_silence_ms=inter_sentence_silence_ms)

        if isinstance(self.model, ChatterboxMultilingualTTS):
            kwargs["language_id"] = language_id or self.language

        audio_prompt_path = kwargs.pop("audio_prompt_path", None)
        if audio_prompt_path and audio_prompt_path != self._last_audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, **kwargs)

        valid_params = set(inspect.signature(self.model.generate).parameters)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        dropped = set(kwargs) - valid_params
        if dropped:
            warnings.warn(
                f"{type(self.model).__name__}.generate() does not accept: {sorted(dropped)}. These kwargs were ignored.",
                stacklevel=2,
            )

        for i, sentence in enumerate(sentences):
            yield self.model.generate(sentence, **filtered_kwargs)
            if silence is not None and i < len(sentences) - 1:
                yield silence

    async def generate_stream_async(
        self,
        text: str,
        language_id: str | None = None,
        normalize_text: bool | None = None,
        inter_sentence_silence_ms: int | None = None,
        **kwargs,
    ) -> AsyncGenerator[torch.Tensor, None]:
        """Async generator yielding one wav tensor per sentence (plus silence tensors between).
        Sentence splitting is always enabled — each yielded chunk corresponds to one sentence.
        Wraps each model.generate() call in asyncio.to_thread() to avoid blocking the event loop."""
        sentences = self._prepare_text(
            text,
            language_id=language_id,
            normalize_text=normalize_text,
            sentence_split=True,
        )
        if not sentences:
            return

        silence = self._silence_chunk(inter_sentence_silence_ms=inter_sentence_silence_ms)

        if isinstance(self.model, ChatterboxMultilingualTTS):
            kwargs["language_id"] = language_id or self.language

        audio_prompt_path = kwargs.pop("audio_prompt_path", None)
        if audio_prompt_path and audio_prompt_path != self._last_audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, **kwargs)

        valid_params = set(inspect.signature(self.model.generate).parameters)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        dropped = set(kwargs) - valid_params
        if dropped:
            warnings.warn(
                f"{type(self.model).__name__}.generate() does not accept: {sorted(dropped)}. These kwargs were ignored.",
                stacklevel=2,
            )

        for i, sentence in enumerate(sentences):
            yield await asyncio.to_thread(self.model.generate, sentence, **filtered_kwargs)
            if silence is not None and i < len(sentences) - 1:
                yield silence

    def generate_stream_fast_sync(
        self,
        text: str,
        language_id: str | None = None,
        normalize_text: bool | None = None,
        inter_sentence_silence_ms: int | None = None,
        **kwargs,
    ) -> Generator[torch.Tensor, None, None]:
        """Sync streaming variant using generate_fast() per sentence.
        Sentence splitting is always enabled — each yielded chunk corresponds to one sentence.
        Falls back to generate_stream_sync() on non-CUDA devices or models without generate_fast."""
        if not hasattr(self.model, "generate_fast"):
            yield from self.generate_stream_sync(
                text,
                language_id=language_id,
                normalize_text=normalize_text,
                inter_sentence_silence_ms=inter_sentence_silence_ms,
                **kwargs,
            )
            return

        sentences = self._prepare_text(
            text,
            language_id=language_id,
            normalize_text=normalize_text,
            sentence_split=True,
        )
        if not sentences:
            return

        silence = self._silence_chunk(inter_sentence_silence_ms=inter_sentence_silence_ms)

        if isinstance(self.model, ChatterboxMultilingualTTS):
            kwargs["language_id"] = language_id or self.language

        audio_prompt_path = kwargs.pop("audio_prompt_path", None)
        if audio_prompt_path and audio_prompt_path != self._last_audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, **kwargs)

        valid_params = set(inspect.signature(self.model.generate_fast).parameters)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        dropped = set(kwargs) - valid_params
        if dropped:
            warnings.warn(
                f"{type(self.model).__name__}.generate_fast() does not accept: {sorted(dropped)}. These kwargs were ignored.",
                stacklevel=2,
            )

        for i, sentence in enumerate(sentences):
            yield self.model.generate_fast(sentence, **filtered_kwargs)
            if silence is not None and i < len(sentences) - 1:
                yield silence

    async def generate_stream_fast_async(
        self,
        text: str,
        language_id: str | None = None,
        normalize_text: bool | None = None,
        inter_sentence_silence_ms: int | None = None,
        **kwargs,
    ) -> AsyncGenerator[torch.Tensor, None]:
        """Async streaming variant using generate_fast() per sentence.
        Sentence splitting is always enabled — each yielded chunk corresponds to one sentence.
        Falls back to generate_stream_async() on non-CUDA devices or models without generate_fast."""
        if not hasattr(self.model, "generate_fast"):
            async for chunk in self.generate_stream_async(
                text,
                language_id=language_id,
                normalize_text=normalize_text,
                inter_sentence_silence_ms=inter_sentence_silence_ms,
                **kwargs,
            ):
                yield chunk
            return

        sentences = self._prepare_text(
            text,
            language_id=language_id,
            normalize_text=normalize_text,
            sentence_split=True,
        )
        if not sentences:
            return

        silence = self._silence_chunk(inter_sentence_silence_ms=inter_sentence_silence_ms)

        if isinstance(self.model, ChatterboxMultilingualTTS):
            kwargs["language_id"] = language_id or self.language

        audio_prompt_path = kwargs.pop("audio_prompt_path", None)
        if audio_prompt_path and audio_prompt_path != self._last_audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, **kwargs)

        valid_params = set(inspect.signature(self.model.generate_fast).parameters)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        dropped = set(kwargs) - valid_params
        if dropped:
            warnings.warn(
                f"{type(self.model).__name__}.generate_fast() does not accept: {sorted(dropped)}. These kwargs were ignored.",
                stacklevel=2,
            )

        for i, sentence in enumerate(sentences):
            yield await asyncio.to_thread(self.model.generate_fast, sentence, **filtered_kwargs)
            if silence is not None and i < len(sentences) - 1:
                yield silence
