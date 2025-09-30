"""Helper that wraps text-to-speech generation for Telegram replies."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import torchaudio
from num2words import num2words

from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES


logger = logging.getLogger(__name__)


NUM2WORDS_LANG = {
    "ar": "ar",
    "da": "da",
    "de": "de",
    "el": "el",
    "en": "en",
    "es": "es",
    "fi": "fi",
    "fr": "fr",
    "he": "he",
    "hi": "hi",
    "it": "it",
    "nl": "nl",
    "no": "no",
    "pl": "pl",
    "pt": "pt",
    "ru": "ru",
    "sv": "sv",
    "tr": "tr",
}

_NUMBER_RE = re.compile(r"\d+")

def _normalize_numbers(text: str, language_id: str) -> str:
    lang = NUM2WORDS_LANG.get(language_id)
    if not lang:
        return text

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        try:
            return num2words(int(token), lang=lang)
        except (ValueError, NotImplementedError):
            return token

    return _NUMBER_RE.sub(replace, text)


class VoiceSynthesizer:
    """Lazily loads the multilingual Chatterbox TTS model and exposes a helper method."""

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        device: str = "cpu",
        language: str = "ru",
    ) -> None:
        self._device = device
        self._model_dir = model_dir
        self._language = language.lower()
        self._tts: Optional[ChatterboxMultilingualTTS] = None

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

        normalized = _normalize_numbers(text, language_id)
        wav = self._tts.generate(normalized, language_id=language_id, cfg_weight=cfg_weight)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), wav, self._tts.sr)
        return output_path

