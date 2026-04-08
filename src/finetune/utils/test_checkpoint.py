#!/usr/bin/env python3
"""Smoke-test a local Chatterbox checkpoint with shared prompt examples.

Example runs:
    python src/finetune/utils/test_checkpoint.py \
        --model_variant base \
        --checkpoint_dir /path/to/checkpoint-50000

    python src/finetune/utils/test_checkpoint.py \
        --model_variant multilingual \
        --checkpoint_dir /path/to/checkpoint-50000 \
        --language_id da \
        --output_dir generation_tests_multilingual

    python src/finetune/utils/test_checkpoint.py \
        --model_variant turbo \
        --checkpoint_dir /path/to/local-model \
        --load_mode full_model \
        --voice_dir ./custom_voices \
        --output_dir generation_tests_turbo
"""

import argparse
import logging
import os
import random
import re
from pathlib import Path

os.environ["PYTORCH_NNPACK"] = "0"
os.environ["USE_NNPACK"] = "0"

import numpy as np
import torch
import torchaudio as ta
from safetensors.torch import load_file

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts import ChatterboxTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROMPTS_FILE = Path(__file__).with_name("text_examples.txt")
FINETUNE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_VOICE_DIR = "voices"


def load_test_prompts() -> list[str]:
    prompts = []
    for raw_line in PROMPTS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(re.sub(r"^\d+\.\s*", "", line))

    if not prompts:
        raise ValueError(f"No prompts found in {PROMPTS_FILE}")

    return prompts


def set_seed(seed: int, device: str):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def resolve_voice_samples(voice_samples: list[str] | None, voice_dir: Path) -> list[str]:
    if voice_samples:
        return voice_samples

    if not voice_dir.exists():
        raise FileNotFoundError(f"Voice directory does not exist: {voice_dir}")

    resolved_samples = sorted(
        str(path) for path in voice_dir.iterdir() if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".flac"}
    )
    if not resolved_samples:
        raise FileNotFoundError(f"No audio samples found in voice directory: {voice_dir}")

    logger.info("Using %s voice sample(s) from %s", len(resolved_samples), voice_dir)
    return resolved_samples


def load_complete_model(model_variant: str, model_dir: Path, device: str):
    if model_variant == "base":
        return ChatterboxTTS.from_local(ckpt_dir=model_dir, device=device)
    if model_variant == "multilingual":
        return ChatterboxMultilingualTTS.from_local(model_dir, device)
    if model_variant == "turbo":
        return ChatterboxTurboTTS.from_local(model_dir, device)
    raise ValueError(f"Unsupported model variant: {model_variant}")


def load_pretrained_model(model_variant: str, device: str):
    device_obj = torch.device(device)

    if model_variant == "base":
        return ChatterboxTTS.from_pretrained(device=device_obj)
    if model_variant == "multilingual":
        return ChatterboxMultilingualTTS.from_pretrained(device_obj)
    if model_variant == "turbo":
        return ChatterboxTurboTTS.from_pretrained(device_obj)
    raise ValueError(f"Unsupported model variant: {model_variant}")


def extract_t3_state_dict(checkpoint_state_dict: dict) -> dict:
    t3_state_dict = {key[3:]: value for key, value in checkpoint_state_dict.items() if key.startswith("t3.")}
    if t3_state_dict:
        return t3_state_dict

    logger.warning("No 't3.'-prefixed weights found; assuming checkpoint already contains bare T3 weights")
    return checkpoint_state_dict


def replace_t3_from_checkpoint(model, model_variant: str, checkpoint_dir: Path):
    model_file = checkpoint_dir / "model.safetensors"
    if not model_file.exists():
        raise FileNotFoundError(f"No model.safetensors found in checkpoint directory: {checkpoint_dir}")

    logger.info("Loading finetuned T3 weights from %s", model_file)
    checkpoint_state_dict = load_file(model_file)
    t3_state_dict = extract_t3_state_dict(checkpoint_state_dict)

    if model_variant == "turbo":
        incompatible_keys = model.t3.load_state_dict(t3_state_dict, strict=False)
        if incompatible_keys.missing_keys:
            logger.warning("Missing turbo T3 keys while loading checkpoint: %s", incompatible_keys.missing_keys)
        if incompatible_keys.unexpected_keys:
            logger.warning("Unexpected turbo T3 keys while loading checkpoint: %s", incompatible_keys.unexpected_keys)
    else:
        model.t3.load_state_dict(t3_state_dict)

    model.t3.eval()


def load_model(
    model_variant: str,
    checkpoint_dir: Path,
    device: str,
    load_mode: str,
):
    if load_mode == "full_model":
        return load_complete_model(model_variant, checkpoint_dir, device)

    model = load_pretrained_model(model_variant, device)
    replace_t3_from_checkpoint(model, model_variant, checkpoint_dir)
    return model


def generate_audio(model, model_variant: str, text: str, voice_sample: Path, language_id: str | None):
    if model_variant == "base":
        return model.generate(
            text,
            audio_prompt_path=voice_sample.as_posix(),
            cfg_weight=0.5,
            min_p=0.1,
            top_p=0.9,
            exaggeration=0.5,
            temperature=0.3,
        )

    if model_variant == "multilingual":
        return model.generate(
            text[:300],
            language_id=language_id,
            audio_prompt_path=voice_sample.as_posix(),
            cfg_weight=0.5,
            exaggeration=0.5,
            temperature=0.3,
        )

    return model.generate(
        text[:300],
        audio_prompt_path=voice_sample.as_posix(),
        temperature=0.8,
    )


def validate_generated_audio(wav_cpu: torch.Tensor) -> bool:
    max_amplitude = wav_cpu.abs().max().item()
    if torch.all(wav_cpu == 0):
        logger.warning("Generated audio is completely silent")
        return False
    if max_amplitude < 1e-6:
        logger.warning("Generated audio is very quiet (max amplitude: %.8f)", max_amplitude)
    else:
        logger.info("Audio appears to have content (max amplitude: %.6f)", max_amplitude)
    return True


def test_checkpoint(
    model_variant: str,
    checkpoint_dir: Path,
    output_dir: Path,
    voice_samples: list[str],
    language_id: str | None,
    seed: int,
    load_mode: str,
) -> bool:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = load_test_prompts()
    set_seed(seed, device)

    if load_mode == "replace_t3":
        logger.info("Loading pretrained %s model and replacing T3 from %s", model_variant, checkpoint_dir)
    else:
        logger.info("Loading %s full model from %s", model_variant, checkpoint_dir)

    model = load_model(
        model_variant=model_variant,
        checkpoint_dir=checkpoint_dir,
        device=device,
        load_mode=load_mode,
    )

    for voice_sample in voice_samples:
        voice_path = Path(voice_sample)
        voice_output_dir = output_dir / voice_path.stem
        voice_output_dir.mkdir(parents=True, exist_ok=True)

        for index, text in enumerate(prompts, start=1):
            logger.info("Testing prompt %s with %s", index, voice_path)
            with torch.inference_mode():
                wav = generate_audio(model, model_variant, text, voice_path, language_id)

            wav_cpu = wav.cpu()
            if not validate_generated_audio(wav_cpu):
                return False

            duration_seconds = wav_cpu.shape[-1] / model.sr
            logger.info("Audio duration: %.2f seconds at %s Hz", duration_seconds, model.sr)
            ta.save(str(voice_output_dir / f"test_{index:02d}.wav"), wav_cpu, model.sr)

    return True


def main():
    parser = argparse.ArgumentParser(description="Test local Chatterbox checkpoints")
    parser.add_argument("--model_variant", choices=["base", "multilingual", "turbo"], required=True)
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        required=True,
        help="Path to a Trainer checkpoint directory or, with --load_mode full_model, a complete local model directory.",
    )
    parser.add_argument(
        "--load_mode",
        choices=["replace_t3", "full_model"],
        default="replace_t3",
        help="Default loads the standard pretrained model and replaces only T3 from checkpoint_dir/model.safetensors. Use full_model only when checkpoint_dir already contains a complete local model.",
    )
    parser.add_argument(
        "--voice_samples",
        type=str,
        nargs="+",
        help="One or more reference audio files. If omitted, files are loaded from --voice_dir.",
    )
    parser.add_argument(
        "--voice_dir",
        type=Path,
        default=DEFAULT_VOICE_DIR,
        help="Directory used when --voice_samples is omitted. Defaults to ./voices.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("generation_tests"),
        help="Directory for generated wav files",
    )
    parser.add_argument("--language_id", type=str, help="Required when model_variant is multilingual")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for reproducible generation")
    args = parser.parse_args()

    if args.model_variant == "multilingual" and not args.language_id:
        args.language_id = "da"
        logger.info("language id defaulting to danish (da)")

    try:
        voice_samples = resolve_voice_samples(args.voice_samples, args.voice_dir)
    except Exception:
        logger.exception("Failed to resolve voice samples")
        return 1

    try:
        success = test_checkpoint(
            model_variant=args.model_variant,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            voice_samples=voice_samples,
            language_id=args.language_id,
            seed=args.seed,
            load_mode=args.load_mode,
        )
    except Exception:
        logger.exception("Checkpoint test failed")
        return 1

    if success:
        logger.info("Checkpoint test passed")
        return 0

    logger.error("Checkpoint test failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
