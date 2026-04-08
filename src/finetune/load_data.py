import logging
from collections.abc import Sized
from functools import partial
from typing import Any

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    VerificationMode,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from datasets.features import Audio
from torchcodec.decoders import AudioDecoder

from finetune.custom_arguments import CustomTrainingArguments, DataArguments, ModelArguments

logger = logging.getLogger(__name__)


# Columns retained after loading for training; all others are dropped.
KEEP_COLUMNS = [
    "audio",
    "text",
    "language",
    "speaker_embedding",
    "cond_prompt_speech_tokens",
]

PREPROCESS_KEEP_COLUMNS = KEEP_COLUMNS + ["speaker_id"]


def normalize_features(dataset, target_sampling_rate: int = 16000):
    """Resample audio column to target sampling rate if it differs."""
    if not isinstance(dataset, Dataset):
        return dataset

    if "audio" in dataset.features:
        current_sampling_rate = dataset.features["audio"].sampling_rate
        if current_sampling_rate != target_sampling_rate:
            logger.info(f"Resampling audio from {current_sampling_rate} Hz to {target_sampling_rate} Hz")
            dataset = dataset.cast_column("audio", Audio(sampling_rate=target_sampling_rate))
    return dataset


def load_data(model_args: ModelArguments, data_args: DataArguments, training_args: CustomTrainingArguments):
    # Download from Hugging Face Hub
    if data_args.datasets:
        train, eval = load_from_hf(data_args, training_args)

    # Load from local disk
    elif data_args.local_hf_dataset:
        logger.info(f"Loading dataset from local disk: {data_args.local_hf_dataset}")
        logger.info(f"Using dataset arguments: {data_args}")
        ds = load_from_disk(data_args.local_hf_dataset)

        if isinstance(ds, DatasetDict):
            if "train" not in ds:
                raise ValueError("Local dataset directory must contain a 'train' split when stored as a DatasetDict.")
            ds = ds["train"]

        if data_args.filter_dataset:
            ds = filter_dataset(
                dataset=ds,
                audio_column="audio",
                min_seconds_per_example=data_args.min_seconds_per_example,
                max_seconds_per_example=data_args.max_seconds_per_example,
                num_proc=None,  # Disable multiprocessing to prevent semaphore leaks
            )

        ds = normalize_features(ds, target_sampling_rate=16000)
        ds = ds.shuffle(seed=training_args.seed)
        logger.info("Dataset loaded successfully.")

        train = ds
        eval = None

        if training_args.do_eval and data_args.eval_split_size > 0 and len(ds) > 1:
            logger.info(f"Splitting dataset for evaluation with ratio {data_args.eval_split_size}")
            split_dataset = ds.train_test_split(test_size=data_args.eval_split_size, seed=training_args.seed)
            train, eval = split_dataset["train"], split_dataset["test"]
            logger.info(f"Evaluation set size: {len(eval)}")

    else:
        return None, None

    return train, eval


def load_single_dataset(
    dataset_config: dict[str, Any],
    data_args: DataArguments,
    verification_mode: VerificationMode,
    keep_columns: list[str],
    shuffle: bool,
    seed: int | None,
):
    dataset_id = dataset_config.get("id")
    if not dataset_id:
        raise ValueError("Each dataset config must include a non-empty 'id'.")

    ds = load_dataset(
        path=dataset_id,
        split=dataset_config.get("split"),
        cache_dir=data_args.dataset_cache_dir,
        verification_mode=verification_mode,
        streaming=data_args.stream_dataset,
        trust_remote_code=True,
        num_proc=None,
    )

    if isinstance(ds, DatasetDict):
        available_splits = [str(split_name) for split_name in ds.keys()]
        logger.info(f"Found DatasetDict with splits: {available_splits}")

        logger.info("Processing splits sequentially to save memory...")
        processed_datasets = []
        for i, split_name in enumerate(available_splits):
            logger.info(f"Processing split {i + 1}/{len(available_splits)}: {split_name}")
            split_ds = ds[split_name]

            processed_datasets.append(split_ds)

        logger.info(f"Concatenating {len(processed_datasets)} splits...")
        ds = concatenate_datasets(processed_datasets)
        logger.info(f"Successfully combined {len(available_splits)} splits")
    elif isinstance(ds, IterableDatasetDict):
        raise ValueError("Preprocessing and finetuning do not support IterableDatasetDict inputs when split is omitted.")

    if not isinstance(ds, (Dataset, IterableDataset)):
        raise TypeError(f"Unsupported dataset type returned from load_dataset: {type(ds)}")

    text_col = dataset_config.get("text_column")
    if text_col and text_col != data_args.text_column_name:
        ds = ds.rename_column(text_col, data_args.text_column_name)

    audio_col = dataset_config.get("audio_column")
    if audio_col and audio_col != data_args.audio_column_name:
        ds = ds.rename_column(audio_col, data_args.audio_column_name)

    speaker_col = dataset_config.get("id_column")
    if speaker_col and speaker_col != "speaker_id" and speaker_col in ds.column_names:
        ds = ds.rename_column(speaker_col, "speaker_id")

    if data_args.filter_dataset or dataset_config.get("filter", False):
        ds = filter_dataset(
            dataset=ds,
            audio_column=data_args.audio_column_name,
            min_seconds_per_example=data_args.min_seconds_per_example,
            max_seconds_per_example=data_args.max_seconds_per_example,
            num_proc=None,
        )

    column_names = getattr(ds, "column_names", []) or []
    if "language" not in column_names:
        language = dataset_config.get("language", "da")
        ds = ds.map(lambda example: {**example, "language": language})

    ds = ds.remove_columns(column_names=[column for column in ds.column_names or list() if column not in keep_columns])

    if shuffle:
        ds = ds.shuffle(seed=seed)

    ds = normalize_features(ds, target_sampling_rate=16000)
    return ds


def load_preprocess_dataset(dataset_config: dict[str, Any], data_args: DataArguments):
    verification_mode = VerificationMode.NO_CHECKS if data_args.ignore_verifications else VerificationMode.BASIC_CHECKS
    return load_single_dataset(
        dataset_config=dataset_config,
        data_args=data_args,
        verification_mode=verification_mode,
        keep_columns=PREPROCESS_KEEP_COLUMNS,
        shuffle=False,
        seed=None,
    )


def load_from_hf(data_args: DataArguments, training_args: CustomTrainingArguments):
    verification_mode = VerificationMode.NO_CHECKS if data_args.ignore_verifications else VerificationMode.BASIC_CHECKS
    dataset_configs = data_args.datasets or {}
    logger.info(f"Loading dataset(s) '{dataset_configs}' from Hugging Face Hub.")
    logger.info(f"Using dataset arguments: {data_args}")

    if not dataset_configs:
        raise ValueError("No datasets configured for Hugging Face loading.")

    all_datasets = []

    for dataset_name, dataset_config in dataset_configs.items():
        logger.info(f"Loading dataset: {dataset_name} with config: {dataset_config}")

        try:
            logger.info(f"Starting dataset loading for {dataset_name}...")
            ds = load_single_dataset(
                dataset_config=dataset_config,
                data_args=data_args,
                verification_mode=verification_mode,
                keep_columns=KEEP_COLUMNS,
                shuffle=True,
                seed=training_args.seed,
            )
            logger.info(f"Successfully loaded dataset {dataset_name}")

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            logger.error(f"Dataset config: {dataset_config}")
            raise RuntimeError(f"Failed to load dataset {dataset_name}: {str(e)}") from e

        all_datasets.append(ds)

    assert len(all_datasets) > 0

    if len(all_datasets) > 1:
        logger.info(f"Concatenating {len(all_datasets)} datasets")
        ds = concatenate_datasets(all_datasets)
    else:
        ds = all_datasets[0]

    train = ds
    eval = None

    if training_args.do_eval and not data_args.stream_dataset:
        logger.info("Splitting dataset for evaluation...")
        if data_args.eval_split_size > 0:
            logger.info(f"Splitting train dataset for evaluation with ratio {data_args.eval_split_size}")
            split_dataset = ds.train_test_split(test_size=data_args.eval_split_size, seed=training_args.seed)
            train, eval = split_dataset["train"], split_dataset["test"]
            logger.info(f"Evaluation set size: {len(eval)}")
    elif training_args.do_eval and data_args.stream_dataset:
        logger.info("Splitting dataset for evaluation...")
        eval = ds.take(500)
        train = ds.skip(500)

    return train, eval


def filter_dataset(
    dataset: Any,
    audio_column: str,
    min_seconds_per_example: float,
    max_seconds_per_example: float | None,
    num_proc: int | None = None,
):
    """Filter the dataset.

    Note that this removes samples from the dataset.

    Args:
        dataset:
            The dataset to filter.
        audio_column:
            The name of the column containing the audio.
        min_seconds_per_example:
            The minimum number of seconds that an example can have.
        max_seconds_per_example:
            The maximum number of seconds that an example can have.
        num_proc (optional):
            The number of processes to use for filtering the dataset. If `None`, then
            no multiprocessing is used. Defaults to `None`.

    Returns:
        The filtered dataset.
    """
    num_samples_before = len(dataset) if isinstance(dataset, Sized) else 0

    filter_fn = partial(
        filter_example,
        audio_column=audio_column,
        min_seconds_per_example=min_seconds_per_example,
        max_seconds_per_example=max_seconds_per_example,
    )
    if isinstance(dataset, (Dataset, DatasetDict)):
        filtered = dataset.filter(function=filter_fn, num_proc=num_proc, desc="Filtering dataset")
    elif isinstance(dataset, IterableDataset):
        filtered = dataset.filter(function=filter_fn)
    else:
        raise TypeError(f"Unsupported dataset type for filtering: {type(dataset)}")

    # Add info back in the filtered dataset, as it gets removed after calling `filter`
    if isinstance(dataset, Dataset):
        filtered.info.features = dataset.info.features
    elif isinstance(dataset, IterableDataset):
        pass
    elif isinstance(dataset, DatasetDict) and isinstance(filtered, DatasetDict):
        for split_name in [str(name) for name in dataset.keys()]:
            filtered[split_name].info.features = dataset[split_name].info.features

    if isinstance(dataset, Sized) and isinstance(filtered, Sized):
        num_samples_removed = num_samples_before - len(filtered)
        logger.info(f"Removed {num_samples_removed:,} samples from the dataset")

    return filtered


def filter_example(
    sample: dict[str, Any],
    audio_column: str,
    min_seconds_per_example: float,
    max_seconds_per_example: float | None,
) -> bool:
    """Filter samples based on the validation status.

    Args:
        sample:
            The sample to filter.
        audio_column:
            The name of the column containing the audio.
        min_seconds_per_example:
            The minimum number of seconds that an example can have.
        max_seconds_per_example:
            The maximum number of seconds that an example can have.

    Returns:
        Whether the sample should be kept.
    """
    audio = sample[audio_column]

    if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
        audio_array = audio["array"]
        sample_rate = audio["sampling_rate"]
    elif isinstance(audio, AudioDecoder):
        try:
            sample_data = audio.get_all_samples()
            audio_array = sample_data.data.cpu().numpy().reshape(-1)
            sample_rate = int(sample_data.sample_rate)
        except Exception as e:
            logger.warning(f"Error decoding audio: {e}")
            return False
    else:
        return False

    if min_seconds_per_example is not None and audio_array.shape[0] < sample_rate * min_seconds_per_example:
        return False

    if max_seconds_per_example is not None and audio_array.shape[0] > sample_rate * max_seconds_per_example:
        return False

    if "validated" in sample and sample["validated"] == "rejected":
        return False

    return True
