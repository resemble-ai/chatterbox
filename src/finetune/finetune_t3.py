"""Based on: https://github.com/stlohrey/chatterbox-finetuning."""

import argparse
import logging
import shutil
import sys
import warnings
from pathlib import Path

import torch
import torch.multiprocessing as mp
from safetensors.torch import save_file
from transformers import EarlyStoppingCallback, HfArgumentParser, Trainer, set_seed

from finetune.custom_arguments import CustomTrainingArguments, DataArguments, ModelArguments
from finetune.custom_models import T3ForFineTuning
from finetune.dataset import SpeechDataCollator, SpeechFineTuningDataset
from finetune.load_data import load_data
from finetune.load_model import load_model

warnings.filterwarnings(
    "ignore",
    message=r"`torch\.backends\.cuda\.sdp_kernel\(\)` is deprecated.*",
    category=FutureWarning,
)

logger = logging.getLogger(__name__)
FINAL_MODEL_SUBDIR = "final_model"


def main():
    logger.info("Starting fine-tuning script...")

    # Parse command line arguments with default config path
    parser = argparse.ArgumentParser(description="Finetune ChatterboxTTS T3 model")
    parser.add_argument(
        "--config",
        type=str,
        default="finetune/configs/finetune_turbo.yaml",
        help="Path to YAML configuration file (default: finetune/configs/finetune_turbo.yaml)",
    )
    args = parser.parse_args()

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / config_path

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    # Load configuration from YAML file
    hf_parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    logger.info(f"Loading configuration from: {config_path}")
    model_args, data_args, training_args = hf_parser.parse_yaml_file(yaml_file=str(config_path))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    if not training_args.do_eval and str(training_args.eval_strategy) != "IntervalStrategy.NO":
        logger.info("do_eval is false; disabling eval_strategy to avoid evaluation during training.")
        training_args.eval_strategy = "no"

    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)
    set_seed(training_args.seed)

    final_model_dir = Path(training_args.output_dir) / FINAL_MODEL_SUBDIR

    logger.info("Loading ChatterboxTTS model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load the model from local directory or Hugging Face Hub
    original_model_dir_for_copy, chatterbox_model = load_model(model_args, training_args, device)

    try:
        t3_model = chatterbox_model.t3
        chatterbox_t3_config_instance = t3_model.hp
    except Exception as e:
        logger.error(f"Failed to access T3 model from loaded checkpoint: {e}")
        sys.exit(1)

    # --- SPECIAL SETTING FOR TURBO ---
    if model_args.model_variant == "turbo":
        logger.info("Turbo Mode: Removing backbone WTE layer...")
        if hasattr(t3_model.tfmr, "wte"):
            del t3_model.tfmr.wte

    if model_args.freeze_voice_encoder:
        for param in chatterbox_model.ve.parameters():
            param.requires_grad = False
        logger.info("Voice Encoder frozen.")
    if model_args.freeze_s3gen:
        for param in chatterbox_model.s3gen.parameters():
            param.requires_grad = False
        logger.info("S3Gen model frozen.")

    for param in t3_model.parameters():
        param.requires_grad = True
    logger.info("T3 model set to trainable.")

    logger.info("Loading and processing dataset...")

    ds_train, ds_eval = load_data(model_args, data_args, training_args)

    if ds_train is None or (training_args.do_eval and ds_eval is None):
        logger.error("Failed to load dataset. Check the dataset configuration.")
        return

    logger.info(f"Dataset loaded of size {len(ds_train)} with eval size {len(ds_eval) if ds_eval else 0}")

    train_dataset = SpeechFineTuningDataset(data_args, chatterbox_model, chatterbox_t3_config_instance, ds_train)

    eval_dataset = None
    if ds_eval and training_args.do_eval:
        eval_dataset = SpeechFineTuningDataset(data_args, chatterbox_model, chatterbox_t3_config_instance, ds_eval)

    data_collator = SpeechDataCollator(
        text_pad_token_id=chatterbox_t3_config_instance.stop_text_token,
        speech_pad_token_id=chatterbox_t3_config_instance.stop_speech_token,
    )

    hf_trainable_model = T3ForFineTuning(t3_model, chatterbox_t3_config_instance)

    if training_args.wandb_project:
        import wandb

        run_name = training_args.run_name or "finetune_t3"
        wandb.init(project=training_args.wandb_project, name=run_name)
        training_args.report_to = ["wandb"]

    callbacks = []
    if training_args.early_stopping_patience is not None and training_args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience))

    # The collator consumes intermediate keys (cond_from_ref, t3_cond_prompt_len) that are
    # not model inputs. HF Trainer >=4.47 wraps the collator with RemoveColumnsCollator,
    # which strips any key not explicitly declared in forward() — including **kwargs entries.
    training_args.remove_unused_columns = False

    trainer = Trainer(
        model=hf_trainable_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    if training_args.label_names is None:
        # Label names returned in the Custom chatterbox model
        trainer.label_names = ["labels_text", "labels_speech"]

    if training_args.do_train:
        logger.info("*** Training T3 model ***")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        final_model_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(final_model_dir))

        logger.info("Saving finetuned T3 model weights for ChatterboxTTS...")
        t3_to_save = trainer.model.t3 if hasattr(trainer.model, "t3") else trainer.model.module.t3
        finetuned_t3_state_dict = t3_to_save.state_dict()

        # Add back tfmr.wte.weight for the Turbo model only (GPT2-based). It was deleted during
        # model setup but the inference loading code expects it to exist before deleting it.
        if model_args.model_variant == "turbo" and "tfmr.wte.weight" not in finetuned_t3_state_dict:
            vocab_size = t3_to_save.cfg.vocab_size
            hidden_size = t3_to_save.cfg.hidden_size
            dummy_wte = torch.nn.Embedding(vocab_size, hidden_size)
            finetuned_t3_state_dict["tfmr.wte.weight"] = dummy_wte.weight.detach().clone()
            logger.info(f"Added dummy tfmr.wte.weight with shape ({vocab_size}, {hidden_size})")

        output_t3_safetensor_path = final_model_dir / "t3_model.safetensors"

        # Find original T3 model filename to use for saving
        if original_model_dir_for_copy:
            original_t3_filename = None
            for src_file in original_model_dir_for_copy.iterdir():
                if src_file.is_file() and src_file.name.startswith("t3") and src_file.suffix == ".safetensors":
                    original_t3_filename = src_file.name
                    logger.info(f"Found original T3 model file: {original_t3_filename}")
                    break

            # Use original filename if found
            if original_t3_filename:
                output_t3_safetensor_path = final_model_dir / original_t3_filename

        save_file(finetuned_t3_state_dict, output_t3_safetensor_path)
        logger.info(f"Finetuned T3 model weights saved to {output_t3_safetensor_path}")

        if original_model_dir_for_copy:
            # Copy all files from the original model directory except T3 model files
            logger.info(f"Copying all files from {original_model_dir_for_copy} to {final_model_dir}")
            for src_file in original_model_dir_for_copy.iterdir():
                if src_file.is_file():
                    # Skip files that start with 't3' and have .safetensors extension
                    if src_file.name.startswith("t3") and src_file.suffix == ".safetensors":
                        logger.info(f"Skipping T3 model file: {src_file.name}")
                        continue
                    dst_file = final_model_dir / src_file.name
                    shutil.copy2(src_file, dst_file)
                    logger.info(f"Copied {src_file.name}")

            logger.info(f"Full model components structured in {final_model_dir}")

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval and eval_dataset:
        logger.info("*** Evaluating T3 model ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("Finetuning script finished.")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
