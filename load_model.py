"""Utility to fetch the Chatterbox multilingual TTS weights if they are not present locally."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from huggingface_hub import snapshot_download


MODEL_REPOSITORY = "ResembleAI/chatterbox"
REQUIRED_FILES: set[str] = {"ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt"}
OPTIONAL_FILES: set[str] = {"Cangjie5_TC.json"}


logger = logging.getLogger(__name__)


def _missing_files(model_dir: Path, required: Iterable[str]) -> list[Path]:
    return [model_dir / name for name in required if not (model_dir / name).exists()]


def missing_required_files(model_dir: Path) -> list[str]:
    """Return a list of required weight filenames missing from model_dir."""
    return [path.name for path in _missing_files(model_dir, REQUIRED_FILES)]


def resolve_model_dir() -> Path:
    """Return the path where TTS weights should live.

    If `AUDIO_MODEL_DIR` is set we respect it. Otherwise we fall back to
    the project-level `models` directory.
    """

    audio_model_dir_env = os.getenv("AUDIO_MODEL_DIR")
    if audio_model_dir_env:
        return Path(audio_model_dir_env)

    if data_dir_env := os.getenv("DATA_DIRECTORY"):
        logger.warning("DATA_DIRECTORY is ignored for model downloads; using project-level 'models' directory.")
    return Path("models")



def ensure_model_present(model_dir: Path) -> None:
    missing = _missing_files(model_dir, REQUIRED_FILES)
    if not missing:
        logger.info("Chatterbox multilingual weights already present in %s", model_dir)
        return

    logger.info("Downloading Chatterbox multilingual TTS weights to %s", model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_REPOSITORY,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        allow_patterns=sorted(REQUIRED_FILES | OPTIONAL_FILES),
    )

    missing_after = _missing_files(model_dir, REQUIRED_FILES)
    if missing_after:
        missing_str = ", ".join(str(path.name) for path in missing_after)
        raise RuntimeError(f"Model download incomplete, missing: {missing_str}")

    logger.info("Chatterbox multilingual TTS weights downloaded successfully")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    load_dotenv()

    model_dir = resolve_model_dir()
    logger.info("Target model directory: %s", model_dir)

    try:
        ensure_model_present(model_dir)
    except Exception:  # pragma: no cover - CLI helper
        logger.exception("Unable to ensure Chatterbox model availability")
        return 1

    print(f"Chatterbox multilingual TTS weights are ready in '{model_dir}'.")
    print("Set AUDIO_MODEL_DIR to this path (or leave unset to use the default resolution in main.py).")
    return 0


if __name__ == "__main__":
    sys.exit(main())





