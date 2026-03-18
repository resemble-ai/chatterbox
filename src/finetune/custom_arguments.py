from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

from transformers import TrainingArguments as HfTrainingArguments


# --- Custom Training Arguments ---
@dataclass
class CustomTrainingArguments(HfTrainingArguments):
    early_stopping_patience: Optional[int] = field(
        default=None, metadata={"help": "Enable early stopping with specified patience. Default: None (disabled)."}
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "WandB project name to log training metrics. Default: None (disabled)."},
    )

# --- Argument Classes (ModelArguments, DataArguments) ---
@dataclass
class ModelArguments:
    model_variant: Literal["base", "multilingual", "turbo"] = field(
        default="base",
        metadata={
            "help": "Specify the model variant for finetuning. Options: 'base', 'multilingual', 'turbo'. Default: 'base'."
        },
    )
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from Hugging Face"}
    )
    local_model_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a local model directory containing the variant-specific T3 weights and companion files. Overrides model_name_or_path for loading."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional Hugging Face cache directory for pretrained model downloads (separate from output_dir/checkpoints)."},
    )
    freeze_voice_encoder: bool = field(default=True, metadata={"help": "Freeze the Voice Encoder."})
    freeze_s3gen: bool = field(default=True, metadata={"help": "Freeze the S3Gen model (speech token to waveform)."})


@dataclass
class DataArguments:
    datasets: Optional[Dict[str, Dict[str, Any]]] = field(
        default=None,
        metadata={"help": "Dictionary of dataset configurations with dataset names as keys and their configs as values."},
    )
    stream_dataset: bool = field(default=False, metadata={"help": "Whether to stream the dataset. Useful for large datasets"})
    local_hf_dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a local Hugging Face dataset directory. If provided, overrides datasets and train_data_files."
        },
    )
    text_column_name: str = field(default="text", metadata={"help": "The name of the text column in the HF dataset."})
    audio_column_name: str = field(default="audio", metadata={"help": "The name of the audio column in the HF dataset."})
    max_text_len: int = field(default=256, metadata={"help": "Maximum length of text tokens (including BOS/EOS)."})
    max_speech_len: int = field(default=800, metadata={"help": "Maximum length of speech tokens (including BOS/EOS)."})
    audio_prompt_duration_s: float = field(
        default=3.0, metadata={"help": "Duration of audio (from start) to use for T3 conditioning prompt tokens (in seconds)."}
    )
    eval_split_size: float = field(
        default=0.0005,
        metadata={
            "help": "Fraction of data to use for evaluation if splitting manually. Not used if datasets provides eval split."
        },
    )
    ignore_verifications: bool = field(default=False, metadata={"help": "Set to true to ignore dataset verifications."})
    filter_dataset: bool = field(default=False, metadata={"help": "Whether to filter the dataset based on audio duration."})
    min_seconds_per_example: float = field(
        default=0.5, metadata={"help": "Minimum audio duration in seconds. Samples shorter than this will be filtered out."}
    )
    max_seconds_per_example: Optional[float] = field(
        default=None, metadata={"help": "Maximum audio duration in seconds. Samples longer than this will be filtered out."}
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the dataset downloaded from huggingface.co"},
    )


@dataclass
class PreprocessArguments:
    seed: int = field(default=42, metadata={"help": "Random seed for reproducible reference selection."})
    preprocess_cache_dir: str = field(
        default="preprocessing_cache",
        metadata={"help": "Directory for preprocess artifacts such as duration and reference-embedding caches."},
    )
    push_to_hub: bool = field(
        default=True,
        metadata={"help": "Push processed datasets to the Hugging Face Hub instead of saving them locally."},
    )
    push_private: bool = field(
        default=False,
        metadata={"help": "Create the Hub dataset as private when push_to_hub is enabled."},
    )
    hub_org: Optional[str] = field(
        default=None,
        metadata={"help": "Optional Hugging Face organization or user prefix for pushed dataset IDs (for example 'my-org')."},
    )
    output_dataset_version: str = field(
        default="_with_embeddings",
        metadata={"help": "Suffix appended to each processed dataset name."},
    )
    local_output_dir: str = field(
        default="processed_datasets",
        metadata={"help": "Directory where processed datasets are saved when push_to_hub is false."},
    )
    embedding_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for reference embedding computation. Increase cautiously to avoid RAM or VRAM issues."},
    )
