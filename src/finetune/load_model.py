import logging
import os
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from transformers import Trainer

logger = logging.getLogger(__name__)
trainer_instance: Optional[Trainer] = None


def load_model(model_args, training_args, device):
    if model_args.model_variant == "multilingual":
        from chatterbox.mtl_tts import REPO_ID
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS as ChatterboxModel
    elif model_args.model_variant == "turbo":
        from chatterbox.tts_turbo import REPO_ID
        from chatterbox.tts_turbo import ChatterboxTurboTTS as ChatterboxModel
    elif model_args.model_variant == "base":
        from chatterbox.tts import REPO_ID
        from chatterbox.tts import ChatterboxTTS as ChatterboxModel
    else:
        logger.error(f"Unknown model variant: {model_args.model_variant}")
        sys.exit(1)

    if model_args.local_model_dir:
        local_dir_path = Path(model_args.local_model_dir)
        logger.info(f"Loading model from local directory: {model_args.local_model_dir}")
        chatterbox_model = ChatterboxModel.from_local(ckpt_dir=local_dir_path, device=device)
        return local_dir_path, chatterbox_model

    elif model_args.model_name_or_path:
        repo_to_download = model_args.model_name_or_path
        logger.info(f"Loading model from Hugging Face Hub: {repo_to_download}")

        cache_root = Path(model_args.cache_dir).expanduser() if model_args.cache_dir else None
        if cache_root:
            cache_root.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using custom model cache directory: {cache_root}")
        else:
            logger.info("Using default Hugging Face cache directory for model download")

        try:
            snapshot_path = snapshot_download(
                repo_id=repo_to_download,
                token=os.getenv("HF_TOKEN") or True,
                # Optional: Filter to download only what you need
                allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
                cache_dir=str(cache_root) if cache_root else None,
            )
        except Exception as e:
            logger.warning(f"Could not download from {repo_to_download}: {e}.")
            sys.exit(f"Error: Failed to load model from local directory: {e}")

        try:
            logger.info("Inserting weight into local model")
            chatterbox_model = ChatterboxModel.from_local(ckpt_dir=snapshot_path, device=device)
            return Path(snapshot_path), chatterbox_model
        except Exception as e:
            logger.error(f"Something went wrong when parsing the model: {e}")
            sys.exit(f"Error: Failed to load model from local directory: {e}")

    else:
        from huggingface_hub.constants import HF_HUB_CACHE

        logger.info(f"Loading model from Hugging Face Hub to default cache location")
        logger.info(f"Expected cache directory: {HF_HUB_CACHE}")
        chatterbox_model = ChatterboxModel.from_pretrained(device=device)
        cache_dir = Path(HF_HUB_CACHE)

        # Model is stored in: hub/models--{org}--{model}/snapshots/{hash}/
        repo_cache_dir = cache_dir / f"models--{REPO_ID.replace('/', '--')}"
        ref_file = repo_cache_dir / "refs" / "main"
        snapshot_hash = ref_file.read_text().strip()
        model_dir = repo_cache_dir / "snapshots" / snapshot_hash

        logger.info(f"Model should be cached at: {model_dir}")
        return model_dir, chatterbox_model
