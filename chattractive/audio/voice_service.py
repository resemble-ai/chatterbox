"""Helper that wraps text-to-speech generation for Telegram replies."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from google import genai

from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES


logger = logging.getLogger(__name__)


_COMBINING_STRESS_CODES = ("\u0300", "\u0301", "\u030f", "\u0341")
_STRESS_APOSTROPHE_RE = re.compile(r"(?<=[\u0400-\u04FF])[\'`\u00B4\u2019](?=[\u0400-\u04FF])")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.!?…])\s+")
_DEFAULT_MAX_CHARS = 220
_DEFAULT_GAP_SECONDS = 0.15


def _apply_fade_out(wav: torch.Tensor, sample_rate: int, fade_seconds: float = 0.15) -> torch.Tensor:
    fade_samples = min(int(sample_rate * fade_seconds), wav.shape[-1])
    if fade_samples <= 0:
        return wav
    faded = wav.clone()
    fade_curve = torch.linspace(1.0, 0.0, fade_samples, device=wav.device, dtype=wav.dtype)
    faded[..., -fade_samples:] = faded[..., -fade_samples:] * fade_curve
    return faded


def _split_text_for_tts(text: str, max_chars: int = _DEFAULT_MAX_CHARS) -> list[str]:
    normalized = ' '.join(text.split())
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(normalized) if s.strip()]
    if not sentences:
        cleaned = normalized.strip()
        return [cleaned] if cleaned else []

    chunks: list[str] = []
    current = sentences[0]
    for sentence in sentences[1:]:
        candidate = f"{current} {sentence}".strip()
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        if chunks and len(current) < max_chars // 3 and len(chunks[-1]) + 1 + len(current) <= max_chars:
            chunks[-1] = f"{chunks[-1]} {current}".strip()
        else:
            chunks.append(current)

    if len(chunks) == 1 and len(chunks[0]) > max_chars:
        words = chunks[0].split()
        chunks = []
        buffer: list[str] = []
        for word in words:
            buffer.append(word)
            candidate = ' '.join(buffer)
            if len(candidate) >= max_chars:
                chunks.append(candidate)
                buffer = []
        if buffer:
            tail = ' '.join(buffer)
            if chunks and len(tail) < max_chars // 3:
                chunks[-1] = f"{chunks[-1]} {tail}".strip()
            else:
                chunks.append(tail)

    return chunks

def _strip_spurious_stress_marks(text: str) -> str:
    for mark in _COMBINING_STRESS_CODES:
        text = text.replace(mark, '')
    return _STRESS_APOSTROPHE_RE.sub('', text)


def _preview(text: str, *, limit: int = 120) -> str:
    if not text:
        return ""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 1]}…"


class VoiceSynthesizer:
    """Lazily loads the multilingual Chatterbox TTS model and exposes a helper method."""

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        device: str = "cpu",
        language: str = "ru",
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-2.0-flash-exp",
    ) -> None:
        self._device = device
        self._model_dir = model_dir
        self._language = language.lower()
        self._tts: Optional[ChatterboxMultilingualTTS] = None
        self._gemini_client: Optional[genai.Client] = None
        self._gemini_model = gemini_model
        voice_path_env = os.getenv("VOICE_PATH")
        self._voice_reference: Optional[Path] = None
        if voice_path_env:
            self._voice_reference = Path(voice_path_env).expanduser()

        if gemini_api_key:
            try:
                self._gemini_client = genai.Client(api_key=gemini_api_key)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception(
                    "Failed to initialize Gemini client for TTS preprocessing: %s",
                    exc,
                )
                self._gemini_client = None

        if self._language not in SUPPORTED_LANGUAGES:
            logger.warning(
                "Unsupported language_id '%s'. Falling back to 'ru'. Supported: %s",
                self._language,
                ", ".join(sorted(SUPPORTED_LANGUAGES)),
            )
            self._language = "ru"

    def _ensure_loaded(self) -> bool:
        if self._tts is not None:
            return True
        if self._model_dir is None:
            logger.warning("Voice synthesizer is disabled: model directory is not set")
            return False
        try:
            self._tts = ChatterboxMultilingualTTS.from_local(self._model_dir, self._device)
            if self._tts.conds is None:
                logger.warning(
                    "Multilingual TTS loaded without built-in voice conditionals; voice responses are disabled"
                )
                self._tts = None
                return False
            if self._voice_reference:
                if not self._voice_reference.is_file():
                    logger.warning("Custom voice reference %s not found; using default conditionals", self._voice_reference)
                else:
                    try:
                        self._tts.prepare_conditionals(str(self._voice_reference), exaggeration=0.5)
                        logger.info("Loaded custom voice conditionals from %s", self._voice_reference)
                    except Exception as exc:
                        logger.warning("Failed to load custom voice %s: %s", self._voice_reference, exc)
            logger.info(
                "Chatterbox multilingual TTS model loaded from %s (language=%s)",
                self._model_dir,
                self._language,
            )
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to load multilingual TTS model: %s", exc)
            self._tts = None
            return False

    def _prepare_text_for_tts(self, text: str, language_id: str) -> str:
        if not self._gemini_client:
            return text

        instructions = (
            "Rewrite the text for Russian speech synthesis."
            " 1) Expand numerals, dates, times, abbreviations, and symbols into Russian words."
            " 2) Replace URLs, emails, and @mentions with short Russian descriptions."
            " 3) Keep punctuation natural for spoken Russian and preserve sentence order."
            " 4) For words not written in Cyrillic (foreign names, brand names, loanwords) provide a Russian phonetic transcription instead of the original spelling."
            " 5) Ensure abbreviations and proper names have the natural Russian stress without using accent marks; always write 'ИТМО' as 'И ТМО', 'ITMO' as 'И ТМО', 'ИИ' as 'И И', and similarly expand other abbreviations so that stresses are unambiguous."
            " 6) Everything you output will be spoken aloud, so do not add explanations about pronunciation—write the final Russian text ready for speech."
            " 7) Do not include formatting symbols, quotation marks, or descriptions; simply replace any non-Cyrillic tokens with their Russian transcription (do not translate English words, only give their phonetic transcription)."
            " 8) For the name 'LISA', always use 'ЛИСА' (not 'ЛИЗА')."
        )
        payload = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            f"{instructions}\n\n"
                            f"Язык ответа: {language_id}\n\n"
                            f"Оригинальный текст:\n{text}"
                        )
                    }
                ],
            }
        ]
        try:
            response = self._gemini_client.models.generate_content(
                model=self._gemini_model,
                contents=payload,
            )
        except Exception as exc:
            logger.warning("Не удалось подготовить текст для озвучки: %s", exc)
            return text

        prepared = getattr(response, "text", None)
        if not prepared:
            logger.warning("Gemini returned an empty normalization result; falling back to raw text")
            return text
        prepared = _strip_spurious_stress_marks(prepared.strip())
        if not prepared:
            logger.warning("Gemini normalization removed all content; using original text")
            return text
        return prepared

    def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        cfg_weight: float = 0.5,
        language: Optional[str] = None,
    ) -> Optional[Path]:
        """Generate speech for *text* and store it to *output_path*.

        The optional *language* argument overrides the default language id for a single call.
        """
        if not text.strip():
            return None
        if not self._ensure_loaded():
            return None

        assert self._tts is not None  # for type checkers

        language_id = (language or self._language).lower()
        if language_id not in SUPPORTED_LANGUAGES:
            logger.warning(
                "Requested unsupported language '%s'; falling back to '%s'",
                language_id,
                self._language,
            )
            language_id = self._language

        prepared_text = self._prepare_text_for_tts(text, language_id)
        if not prepared_text.strip():
            logger.warning("Подготовленный текст для озвучки пуст; используем оригинальный ответ")
            prepared_text = text.strip()

        logger.debug("Gemini reply preview: %s", _preview(text))
        logger.debug("Gemini TTS preview: %s", _preview(prepared_text))

        chunks = _split_text_for_tts(prepared_text)
        if not chunks:
            logger.warning("Text splitting yielded no segments; skipping voice synthesis")
            return None

        wav_segments = []
        for idx, chunk in enumerate(chunks, start=1):
            logger.debug("Generating chunk %s/%s: %s", idx, len(chunks), chunk)
            segment = self._tts.generate(chunk, language_id=language_id, cfg_weight=cfg_weight)
            if segment.dim() == 1:
                segment = segment.unsqueeze(0)
            wav_segments.append(segment)

        if len(wav_segments) == 1:
            wav = _apply_fade_out(wav_segments[0], self._tts.sr, fade_seconds=_DEFAULT_GAP_SECONDS * 2)
        else:
            gap_samples = max(1, int(self._tts.sr * _DEFAULT_GAP_SECONDS))
            silence = torch.zeros((1, gap_samples), dtype=wav_segments[0].dtype, device=wav_segments[0].device)
            pieces = []
            for idx, segment in enumerate(wav_segments):
                pieces.append(segment)
                if idx + 1 < len(wav_segments):
                    pieces.append(silence)
            wav = torch.cat(pieces, dim=1)

        wav = _apply_fade_out(wav, self._tts.sr, fade_seconds=_DEFAULT_GAP_SECONDS * 2)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), wav, self._tts.sr)
        return output_path
