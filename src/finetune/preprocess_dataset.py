"""Precompute speaker conditioning features for finetuning datasets.

Run with the built-in preprocessing config:

`python -m finetune.preprocess_dataset`

Each input dataset must expose a speaker identifier column so references can be
selected from the same speaker. By default that column is expected to be named
`speaker_id`, but you can remap another source column with `id_column` inside
each dataset config.

When `push_to_hub` is true, the processed dataset is uploaded to the Hub and is
not saved locally. When `push_to_hub` is false, the processed dataset is saved
under `local_output_dir` instead. Warning: local saves create an additional copy
of the dataset on disk and can significantly increase storage usage.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np
import torch
from datasets import Sequence, Value
from huggingface_hub import HfApi
from transformers import HfArgumentParser

from chatterbox.models.s3tokenizer import S3_SR
from finetune.custom_arguments import DataArguments, ModelArguments, PreprocessArguments
from finetune.load_data import load_preprocess_dataset
from finetune.load_model import load_model


def get_audio_duration(audio_array: np.ndarray, sampling_rate: int) -> float:
    """Calculate audio duration in seconds."""
    return len(audio_array) / sampling_rate


def compute_all_durations(dataset, cache_file: str) -> Dict[int, float]:
    """Pre-compute all audio durations once using efficient batch access."""
    total = len(dataset)
    durations = {}
    if Path(cache_file).exists():
        print(f"Loading durations from cache: {cache_file}")
        try:
            with open(cache_file, "r") as f:
                durations_str_keys = json.load(f)
            durations = {int(k): v for k, v in durations_str_keys.items()}
            print(f"Loaded {len(durations)} cached durations")

            if len(durations) == total:
                print("Cache is complete! No recomputation needed.")
                return durations

            print(f"Found partial cache ({len(durations)}/{total} samples)")
            print("Resuming from checkpoint...")
        except Exception as e:
            print(f"Error loading cache: {e}")
            print("Starting fresh computation...")
            durations = {}
    else:
        print(f"Creating initial cache file: {cache_file}")
        try:
            with open(cache_file, "w") as f:
                json.dump({}, f)
        except Exception as e:
            print(f"Warning: Could not create initial cache file: {e}")

    if durations:
        print(f"Skipping {len(durations)} already computed durations")

    print("Pre-computing audio durations...")
    batch_size = 1000

    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch_indices = range(start_idx, end_idx)
        indices_to_compute = [idx for idx in batch_indices if idx not in durations]

        if not indices_to_compute:
            continue

        if start_idx % 10000 == 0:
            print(f"Computing durations: {len(durations)}/{total} ({100 * len(durations) / total:.1f}% complete)")

        batch = dataset[start_idx:end_idx]
        audio_batch = batch["audio"]

        for i, audio_data in enumerate(audio_batch):
            idx = start_idx + i
            if idx in durations:
                continue
            durations[idx] = get_audio_duration(audio_data["array"], audio_data["sampling_rate"])

        if cache_file and len(durations) % 5000 == 0 and len(durations) > 0:
            print(f"Checkpoint: Saving progress to {cache_file} ({len(durations)}/{total} samples)...")
            try:
                with open(cache_file, "w") as f:
                    json.dump(durations, f)
            except Exception as e:
                print(f"Warning: Could not save progress: {e}")

    print(f"Computed durations for {len(durations)} samples")

    if cache_file:
        print(f"Saving durations cache to {cache_file}")
        try:
            with open(cache_file, "w") as f:
                json.dump(durations, f)
            print("Cache saved successfully")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    return durations


def group_by_speaker(dataset) -> Dict[str, List[int]]:
    """Group dataset indices by speaker_id using efficient batch or column access."""
    print("Grouping dataset by speaker...")
    total = len(dataset)

    try:
        all_speaker_ids = dataset["speaker_id"]
        print(f"Retrieved {len(all_speaker_ids)} speaker IDs in batch mode")

        speaker_to_indices = defaultdict(list)
        for idx, speaker_id in enumerate(all_speaker_ids):
            speaker_to_indices[speaker_id].append(idx)

        return dict(speaker_to_indices)
    except Exception as e:
        print(f"Batch column access failed ({e}), falling back to batched iteration...")

    speaker_to_indices = defaultdict(list)
    batch_size = 10000

    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        if start_idx % 50000 == 0:
            print(f"Grouping by speaker: {start_idx}/{total}")

        batch = dataset[start_idx:end_idx]
        speaker_ids = batch["speaker_id"]

        for i, speaker_id in enumerate(speaker_ids):
            speaker_to_indices[speaker_id].append(start_idx + i)

    return dict(speaker_to_indices)


def find_valid_references_for_speaker(
    speaker_indices: List[int], durations: Dict[int, float], min_duration: float = 5.0
) -> List[int]:
    """Find valid reference clips for a speaker based on pre-computed durations."""
    return [idx for idx in speaker_indices if durations[idx] >= min_duration]


def precompute_speaker_valid_refs(
    speaker_to_indices: Dict[str, List[int]],
    durations: Dict[int, float],
    min_duration: float = 5.0,
) -> Dict[str, List[int]]:
    """Pre-compute valid reference indices for each speaker."""
    print("Pre-computing valid references per speaker...")
    speaker_valid_refs = {}

    for speaker_id, indices in speaker_to_indices.items():
        speaker_valid_refs[speaker_id] = find_valid_references_for_speaker(indices, durations, min_duration)

    speakers_with_valid = sum(1 for refs in speaker_valid_refs.values() if refs)
    speakers_without_valid = len(speaker_to_indices) - speakers_with_valid

    print("Speaker reference statistics:")
    print(f"  - Speakers with valid single-clip references: {speakers_with_valid}")
    print(f"  - Speakers without valid references: {speakers_without_valid}")

    return speaker_valid_refs


def prepare_audio_for_reference(audio_data) -> np.ndarray:
    wav_array = audio_data["array"]
    sr = audio_data["sampling_rate"]

    if sr != S3_SR:
        wav_array = librosa.resample(wav_array, orig_sr=sr, target_sr=S3_SR)

    return wav_array.astype(np.float32)


def precompute_reference_embeddings(
    dataset,
    voice_encoder,
    speech_tokenizer,
    speech_cond_prompt_len: int,
    speaker_valid_refs: Dict[str, List[int]],
    batch_size: int,
    cache_file: Optional[str] = None,
) -> Dict[int, Dict[str, List]]:
    """Pre-compute embeddings for all valid reference clips."""
    all_valid_refs = set()
    for refs in speaker_valid_refs.values():
        all_valid_refs.update(refs)

    print(f"Pre-computing embeddings for {len(all_valid_refs)} unique valid references...")

    ref_embeddings = {}
    if cache_file and Path(cache_file).exists():
        try:
            with open(cache_file, "r") as f:
                cached = json.load(f)
            ref_embeddings = {int(k): v for k, v in cached.items()}
            print(f"Loaded {len(ref_embeddings)} cached reference embeddings")

            missing = all_valid_refs - set(ref_embeddings.keys())
            if not missing:
                print("Cache is complete!")
                return ref_embeddings
            print(f"Need to compute {len(missing)} more embeddings")
        except Exception as e:
            print(f"Could not load cache: {e}")

    pending_refs = [ref_idx for ref_idx in sorted(all_valid_refs) if ref_idx not in ref_embeddings]
    effective_batch_size = max(1, batch_size)

    for start_idx in range(0, len(pending_refs), effective_batch_size):
        batch_ref_indices = pending_refs[start_idx : start_idx + effective_batch_size]

        if start_idx % 500 == 0:
            print(
                f"Computing reference embeddings: {start_idx}/{len(pending_refs)} "
                f"({100 * start_idx / max(1, len(pending_refs)):.1f}%)"
            )

        wav_batch = []
        valid_batch_indices = []

        for ref_idx in batch_ref_indices:
            try:
                wav_batch.append(prepare_audio_for_reference(dataset[ref_idx]["audio"]))
                valid_batch_indices.append(ref_idx)
            except Exception as e:
                print(f"Warning: Failed to load audio for ref {ref_idx}: {e}")

        if not wav_batch:
            continue

        try:
            speaker_emb_batch = voice_encoder.embeds_from_wavs(wav_batch, sample_rate=S3_SR)
            tokens_batch, _ = speech_tokenizer.forward(wav_batch, max_len=speech_cond_prompt_len)
        except Exception as e:
            print(f"Warning: Failed to compute batched embeddings for refs {valid_batch_indices}: {e}")
            continue

        for batch_pos, ref_idx in enumerate(valid_batch_indices):
            speaker_emb_list = speaker_emb_batch[batch_pos].tolist()

            if tokens_batch is None:
                cond_tokens = [0] * speech_cond_prompt_len
            else:
                tokens = tokens_batch[batch_pos].cpu().tolist()
                if len(tokens) < speech_cond_prompt_len:
                    tokens = tokens + [0] * (speech_cond_prompt_len - len(tokens))
                elif len(tokens) > speech_cond_prompt_len:
                    tokens = tokens[:speech_cond_prompt_len]
                cond_tokens = tokens

            ref_embeddings[ref_idx] = {
                "speaker_embedding": speaker_emb_list,
                "cond_prompt_speech_tokens": cond_tokens,
            }

        if cache_file and len(ref_embeddings) % 1000 == 0:
            with open(cache_file, "w") as f:
                json.dump(ref_embeddings, f)

    if cache_file:
        print(f"Saving reference embeddings cache to {cache_file}")
        with open(cache_file, "w") as f:
            json.dump(ref_embeddings, f)

    print(f"Computed embeddings for {len(ref_embeddings)} references")
    return ref_embeddings


def create_embedding_mapper(
    ref_embeddings: Dict[int, Dict[str, List]],
    speaker_valid_refs: Dict[str, List[int]],
    random_seed: int,
    speech_cond_prompt_len: int,
):
    """Create a lookup-based mapper using pre-computed embeddings."""

    def assign_embeddings(example, idx):
        rng = np.random.default_rng(random_seed + idx)
        speaker_id = example["speaker_id"]
        valid_refs = speaker_valid_refs.get(speaker_id, [])

        ref_idx = None

        if valid_refs:
            available_refs = [ref for ref in valid_refs if ref != idx and ref in ref_embeddings]
            if available_refs:
                ref_idx = int(rng.choice(available_refs))

        if ref_idx is None and valid_refs:
            available_refs = [ref for ref in valid_refs if ref in ref_embeddings]
            if available_refs:
                ref_idx = int(rng.choice(available_refs))

        if ref_idx is None and idx in ref_embeddings:
            ref_idx = idx

        if ref_idx is not None and ref_idx in ref_embeddings:
            example["speaker_embedding"] = ref_embeddings[ref_idx]["speaker_embedding"]
            example["cond_prompt_speech_tokens"] = ref_embeddings[ref_idx]["cond_prompt_speech_tokens"]
        else:
            example["speaker_embedding"] = [0.0] * 256
            example["cond_prompt_speech_tokens"] = [0] * speech_cond_prompt_len
            print(f"Warning: No valid reference for idx {idx}, speaker {speaker_id}")

        return example

    return assign_embeddings


def add_speaker_embeddings(
    dataset,
    voice_encoder,
    speech_tokenizer,
    speech_cond_prompt_len: int,
    dataset_name: str,
    preprocess_cache_dir: str = "preprocessing_cache",
    min_duration: float = 5.0,
    random_seed: int = 42,
    embedding_batch_size: int = 4,
):
    """Add speaker conditioning columns to a dataset."""
    cache_dir = Path(preprocess_cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    duration_cache_file = cache_dir / f"{dataset_name}_durations_cache.json"
    durations = compute_all_durations(dataset, cache_file=str(duration_cache_file))

    speaker_to_indices = group_by_speaker(dataset)
    print(f"Found {len(speaker_to_indices)} unique speakers")

    speaker_valid_refs = precompute_speaker_valid_refs(speaker_to_indices, durations, min_duration)

    embeddings_cache_file = cache_dir / f"{dataset_name}_ref_embeddings_cache.json"
    ref_embeddings = precompute_reference_embeddings(
        dataset=dataset,
        voice_encoder=voice_encoder,
        speech_tokenizer=speech_tokenizer,
        speech_cond_prompt_len=speech_cond_prompt_len,
        speaker_valid_refs=speaker_valid_refs,
        batch_size=embedding_batch_size,
        cache_file=str(embeddings_cache_file),
    )

    mapper_fn = create_embedding_mapper(
        ref_embeddings=ref_embeddings,
        speaker_valid_refs=speaker_valid_refs,
        random_seed=random_seed,
        speech_cond_prompt_len=speech_cond_prompt_len,
    )

    print("\nAssigning pre-computed embeddings via map...")
    print(f"Dataset has {len(dataset)} samples")

    new_dataset = dataset.map(
        mapper_fn,
        with_indices=True,
        desc="Assigning embeddings",
        num_proc=None,
    )

    print(f"Map completed! Dataset has {len(new_dataset)} samples")

    new_dataset = new_dataset.cast_column("speaker_embedding", Sequence(feature=Value("float32"), length=256))
    new_dataset = new_dataset.cast_column(
        "cond_prompt_speech_tokens",
        Sequence(feature=Value("int64"), length=speech_cond_prompt_len),
    )

    return new_dataset


def dataset_exists_on_hub(dataset_id: str) -> bool:
    """Check if a dataset already exists on Hugging Face Hub."""
    api = HfApi()
    try:
        api.dataset_info(dataset_id)
        return True
    except Exception:
        return False


def build_hub_dataset_name(dataset_name: str, preprocess_args: PreprocessArguments) -> str:
    output_name = f"{dataset_name}{preprocess_args.output_dataset_version}"
    hub_org = (preprocess_args.hub_org or "").strip().strip("/")
    if hub_org:
        return f"{hub_org}/{output_name}"
    return output_name


def get_config_path(config: str) -> Path:
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / config_path

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    return config_path


def main():
    config_path = get_config_path("finetune/configs/preprocess_config.yaml")

    hf_parser = HfArgumentParser((ModelArguments, DataArguments, PreprocessArguments))
    print(f"Loading configuration from: {config_path}")
    model_args, data_args, preprocess_args = hf_parser.parse_yaml_file(yaml_file=str(config_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if data_args.stream_dataset:
        raise ValueError("Preprocessing requires random-access datasets, so stream_dataset must be false.")

    random_seed = preprocess_args.seed if preprocess_args.seed is not None else 42
    min_duration = 5.0 if model_args.model_variant == "turbo" else 3.0
    print(f"Using TTS model: {model_args.model_variant}, minimum reference duration: {min_duration} seconds")
    print(f"Reference embedding batch size: {max(1, preprocess_args.embedding_batch_size)}")

    print(f"\nLoading Chatterbox model on {device}...")
    _, chatterbox_model = load_model(model_args, preprocess_args, device)

    voice_encoder = chatterbox_model.ve
    speech_tokenizer = chatterbox_model.s3gen.tokenizer
    speech_cond_prompt_len = chatterbox_model.t3.hp.speech_cond_prompt_len

    print(f"Speech cond prompt length: {speech_cond_prompt_len} tokens")

    if not data_args.datasets:
        raise ValueError("No datasets configured for preprocessing.")

    for dataset_name, dataset_config in data_args.datasets.items():
        output_dataset_name = build_hub_dataset_name(dataset_name, preprocess_args)

        if preprocess_args.push_to_hub and dataset_exists_on_hub(output_dataset_name):
            print(f"\nDataset '{output_dataset_name}' already exists on Hub. Skipping...")
            continue

        print(f"\nLoading dataset: {dataset_name}")
        dataset = load_preprocess_dataset(dataset_config, data_args)
        dataset_columns = dataset.column_names or []

        if "speaker_id" not in dataset_columns:
            raise ValueError(
                f"Dataset '{dataset_name}' is missing 'speaker_id'. Set 'id_column' in the config to remap the speaker column."
            )

        print("\nDataset info:")
        print(f"- Columns: {dataset_columns}")
        print(f"- Number of samples: {len(dataset)}")

        processed_dataset = add_speaker_embeddings(
            dataset,
            dataset_name=dataset_name,
            voice_encoder=voice_encoder,
            speech_tokenizer=speech_tokenizer,
            speech_cond_prompt_len=speech_cond_prompt_len,
            preprocess_cache_dir=preprocess_args.preprocess_cache_dir,
            min_duration=min_duration,
            random_seed=random_seed,
            embedding_batch_size=preprocess_args.embedding_batch_size,
        )

        print("\nNew dataset columns:", processed_dataset.column_names)

        example = processed_dataset[0]
        print("\nExample entry:")
        print(f"- Text: {example['text'][:100]}...")
        print(f"- Speaker ID: {example['speaker_id']}")
        print(f"- Audio duration: {len(example['audio']['array']) / example['audio']['sampling_rate']:.2f}s")
        print(f"- Speaker embedding shape: {len(example['speaker_embedding'])} (256-dim vector)")
        print(f"- Tokens shape: {len(example['cond_prompt_speech_tokens'])} ({speech_cond_prompt_len} tokens)")

        if preprocess_args.push_to_hub:
            print(f"\nPushing dataset to Hugging Face Hub: {output_dataset_name} ...")
            processed_dataset.push_to_hub(output_dataset_name, private=preprocess_args.push_private)
        else:
            local_output_dir = Path(preprocess_args.local_output_dir).expanduser()
            local_output_dir.mkdir(parents=True, exist_ok=True)
            output_path = local_output_dir / f"{dataset_name}{preprocess_args.output_dataset_version}"
            print("\nSaving processed dataset locally...")
            print("Warning: local save creates an additional dataset copy and may significantly increase disk usage.")
            print(f"Output path: {output_path}")
            processed_dataset.save_to_disk(str(output_path))

    print("\nDone!")


if __name__ == "__main__":
    main()
