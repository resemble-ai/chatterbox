"""Helper that wraps text-to-speech generation for Telegram replies."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torchaudio

from .tts import ChatterboxTTS


logger = logging.getLogger(__name__)


class VoiceSynthesizer:
    """Lazily loads the Chatterbox TTS model and exposes a helper method."""

    def __init__(self, model_dir: Optional[Path] = None, device: str = "cpu") -> None:
        self._device = device
        self._tts: Optional[ChatterboxTTS] = None
        self._model_dir = model_dir

    def _ensure_loaded(self) -> bool:
        if self._tts is not None:
            return True
        if self._model_dir is None:
            logger.warning("Voice synthesizer is disabled: model directory is not set")
            return False
        try:
            self._tts = ChatterboxTTS.from_local(self._model_dir, self._device)
            logger.info("Chatterbox TTS model loaded from %s", self._model_dir)
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to load TTS model: %s", exc)
            self._tts = None
            return False

    def synthesize(self, text: str, output_path: Path, *, cfg_weight: float = 0.5) -> Optional[Path]:
        if not text.strip():
            return None
        if not self._ensure_loaded():
            return None
        assert self._tts is not None  # for type checkers
        wav = self._tts.generate(text, cfg_weight=cfg_weight)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), wav, self._tts.sr)
        return output_path
