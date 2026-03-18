#!/usr/bin/env python3
"""Convert training checkpoints into a local `from_local()` model directory.

Example runs:
    python src/finetune/utils/convert_checkpoint.py \
        /path/to/checkpoint-50000 \
        /path/to/original-model-dir \
        --model_variant base

    python src/finetune/utils/convert_checkpoint.py \
        /path/to/checkpoint-50000 \
        /path/to/original-model-dir \
        --model_variant multilingual \
        --output_dir /path/to/converted-model

    python src/finetune/utils/convert_checkpoint.py \
        /path/to/checkpoint-50000 \
        /path/to/original-model-dir \
        --model_variant turbo \
        --all \
        --output_dir /path/to/converted-checkpoints
"""

import argparse
import logging
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T3_FILENAMES = {
    "base": "t3_cfg.safetensors",
    "multilingual": "t3_mtl23ls_v2.safetensors",
    "turbo": "t3_turbo_v1.safetensors",
}


def extract_t3_state_dict(checkpoint_state_dict: dict) -> dict:
    t3_state_dict = {key[3:]: value for key, value in checkpoint_state_dict.items() if key.startswith("t3.")}
    if t3_state_dict:
        return t3_state_dict

    logger.warning("No 't3.'-prefixed weights found; assuming checkpoint already contains bare T3 weights")
    return checkpoint_state_dict


def convert_checkpoint_to_chatterbox_format(
    checkpoint_dir: Path,
    original_model_dir: Path,
    model_variant: str,
    output_dir: Path | None = None,
) -> Path:
    if output_dir is None:
        output_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}_chatterbox"

    if output_dir.resolve() == original_model_dir.resolve():
        raise ValueError("output_dir must be different from original_model_dir")

    model_file = checkpoint_dir / "model.safetensors"
    if not model_file.exists():
        raise FileNotFoundError(f"No model.safetensors found in {checkpoint_dir}")

    logger.info("Copying model assets from %s to %s", original_model_dir, output_dir)
    shutil.copytree(original_model_dir, output_dir, dirs_exist_ok=True)

    logger.info("Loading checkpoint state dict from %s", model_file)
    checkpoint_state_dict = load_file(model_file)
    t3_state_dict = extract_t3_state_dict(checkpoint_state_dict)

    t3_output_path = output_dir / T3_FILENAMES[model_variant]
    save_file(t3_state_dict, t3_output_path)
    logger.info("Saved %s weights to %s", model_variant, t3_output_path)
    logger.info("Converted checkpoint available at %s", output_dir)
    return output_dir


def iter_checkpoint_dirs(checkpoint_dir: Path) -> list[Path]:
    parent_dir = checkpoint_dir.parent
    checkpoint_dirs = sorted(path for path in parent_dir.glob("checkpoint-*") if path.is_dir())
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint-* directories found in {parent_dir}")
    return checkpoint_dirs


def main():
    parser = argparse.ArgumentParser(description="Convert training checkpoints to Chatterbox local model format")
    parser.add_argument("checkpoint_dir", type=Path, help="Path to a training checkpoint directory")
    parser.add_argument("original_model_dir", type=Path, help="Path to the original local model directory")
    parser.add_argument(
        "--model_variant",
        choices=sorted(T3_FILENAMES),
        required=True,
        help="Model variant that decides which T3 filename to write",
    )
    parser.add_argument("--output_dir", type=Path, help="Output directory or output root when used with --all")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert every checkpoint-* directory next to checkpoint_dir",
    )
    args = parser.parse_args()

    if not args.checkpoint_dir.exists():
        logger.error("Checkpoint directory does not exist: %s", args.checkpoint_dir)
        return 1

    if not args.original_model_dir.exists():
        logger.error("Original model directory does not exist: %s", args.original_model_dir)
        return 1

    try:
        if args.all:
            checkpoint_dirs = iter_checkpoint_dirs(args.checkpoint_dir)
            logger.info("Found %s checkpoints to convert", len(checkpoint_dirs))

            for checkpoint_dir in checkpoint_dirs:
                output_dir = None
                if args.output_dir is not None:
                    output_dir = args.output_dir / f"{checkpoint_dir.name}_chatterbox"
                convert_checkpoint_to_chatterbox_format(
                    checkpoint_dir=checkpoint_dir,
                    original_model_dir=args.original_model_dir,
                    model_variant=args.model_variant,
                    output_dir=output_dir,
                )
        else:
            convert_checkpoint_to_chatterbox_format(
                checkpoint_dir=args.checkpoint_dir,
                original_model_dir=args.original_model_dir,
                model_variant=args.model_variant,
                output_dir=args.output_dir,
            )
    except Exception as exc:
        logger.error("Checkpoint conversion failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
