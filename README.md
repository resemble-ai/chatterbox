# CoRal Chatterbox TTS

A fork of [Resemble AI's Chatterbox TTS](https://github.com/resemble-ai/chatterbox), extended by the [Alexandra Institute](https://www.alexandra.dk/) as part of the [CoRal project](https://huggingface.co/CoRal-project) with a finetuning framework and optimised inference targeted towards Danish language support, though it can easily be adapted for other languages.

The finetuning framework supports all three Chatterbox model variants (base, multilingual, turbo). The original Chatterbox inference code is preserved under `src/chatterbox/`, and a thin blocking inference wrapper adds text normalization, sentence splitting, and custom checkpoint loading without changing the core finetuning path. Finetuning scripts inspired by [stlohrey's chatterbox-finetuning](https://github.com/stlohrey/chatterbox-finetuning).

- [Installation](#installation)
- [Setup](#setup)
- [Finetuning](#finetuning)
- [Inference](#inference)
- [License](#license)

## Installation

Requires Python 3.10 or later. Tested on Python 3.11.

We recommend [uv](https://docs.astral.sh/uv/) as the package manager:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.11 (if not already available)
uv python install 3.11

# Install base dependencies
uv sync

# With finetuning dependencies
uv sync --extra finetune

# With multilingual dependencies
uv sync --extra multilingual
```

Alternatively, with pip:

```bash
pip install -e .
pip install -e ".[finetune]"
pip install -e ".[multilingual]"
```

### CUDA Compatibility

The project pins `torch==2.7.1` and `torchaudio==2.7.1` to support CUDA 12.8 used for training. If another CUDA version is required, modify the versions in `pyproject.toml`. 

Specific wheels may also be required depending on the CUDA version. See the [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/#using-a-pytorch-index) for details on installing specific CUDA builds.



## Setup

Log in to Hugging Face (required to download models and upload finetuned models) and Weights & Biases (optional, for training metric logging):

```bash
uv run huggingface-cli login
uv run wandb login
```

## Finetuning

### Overview

The finetuning workflow consists of the following steps:

1. **Preprocess** (optional) -- Precompute speaker embeddings and conditioning tokens for your dataset
2. **Train** -- Finetune the T3 language model component while keeping other components frozen
3. **Evaluate** -- Test the finetuned checkpoint with reference audio samples
4. **Convert/Upload** -- Convert checkpoints to standalone model directories and upload to Hugging Face

All scripts are run from the `src/` directory using Python module syntax:

```bash
cd src
python -m finetune.finetune_t3
python -m finetune.preprocess_dataset
python -m finetune.hyperparam_search
python -m finetune.utils.test_checkpoint --help
python -m finetune.utils.convert_checkpoint --help
```

### Configuration

Training is driven by YAML config files in `src/finetune/configs/`:

| Config | Variant | Description |
|---|---|---|
| `finetune_base.yaml` | base | English base model finetuning |
| `finetune_mtl.yaml` | multilingual | Multilingual model finetuning |
| `finetune_turbo.yaml` | turbo | Turbo model finetuning |
| `finetune_test.yaml` | multilingual | Quick test run (1000 steps, no saving) |
| `preprocess_config.yaml` | -- | Dataset preprocessing settings |

Key config sections:

**Dataset configuration** -- supports multiple datasets with per-dataset column mapping:

```yaml
datasets:
  my_dataset:
    id: my-org/my-dataset          # Hugging Face dataset ID
    split: null                     # Dataset split (null = all)
    text_column: text               # Column containing transcriptions
    audio_column: audio             # Column containing audio
    language: da                    # Language code
    filter: true                    # true if dataset lacks pre-computed embeddings
```

**Model arguments:**

| Field | Description |
|---|---|
| `model_variant` | `"base"`, `"multilingual"`, or `"turbo"` |
| `local_model_dir` | Path to a local model directory (overrides Hub download) |
| `output_dir` | Where checkpoints and `final_model/` are written |
| `cache_dir` | Hugging Face cache directory for model downloads |

**Training arguments** -- extends HuggingFace `TrainingArguments`:

| Field | Description |
|---|---|
| `learning_rate` | Learning rate (varies by variant, see config defaults) |
| `num_train_epochs` | Number of training epochs |
| `per_device_train_batch_size` | Batch size per device |
| `gradient_accumulation_steps` | Gradient accumulation steps |
| `max_grad_norm` | Gradient clipping threshold |
| `warmup_steps` | Learning rate warmup steps |
| `wandb_project` | WandB project name (enables logging) |
| `early_stopping_patience` | Early stopping patience (null = disabled) |

For the complete list of arguments, see `src/finetune/custom_arguments.py`.

### Dataset Preprocessing (Optional)

Preprocessing is **optional but recommended**. It precomputes speaker embeddings and speech conditioning tokens from other utterances of the same speaker, which improves voice quality and consistency.

**Without preprocessing:** The training script can compute conditioning directly from each training audio clip by masking the first portion of the audio. This requires that each sample is long enough (e.g. at least 5 seconds for turbo, 3 seconds for base) -- set `filter: true` and configure `min_seconds_per_example` in the dataset config to enforce this.

**With preprocessing:** Speaker embeddings are computed from a separate reference utterance of the same speaker, allowing shorter training clips and better speaker conditioning. This requires a `speaker_id` column in the dataset (or configure `id_column` to remap from another column name).

The training script supports mixing preprocessed and non-preprocessed datasets in the same run. In the dataset config, set `filter: false` for preprocessed datasets (embeddings already present) and `filter: true` for non-preprocessed datasets (embeddings computed on-the-fly, with length filtering applied).

To run preprocessing:

```bash
cd src
python -m finetune.preprocess_dataset
```

Configuration is in `src/finetune/configs/preprocess_config.yaml`. Key options:

| Field | Description |
|---|---|
| `model_variant` | Which model's encoder to use for embeddings |
| `push_to_hub` | Upload processed dataset to Hugging Face Hub |
| `push_private` | Make the uploaded dataset private |
| `hub_org` | Hugging Face organization for the uploaded dataset |
| `embedding_batch_size` | Batch size for embedding computation (increase cautiously) |
| `local_output_dir` | Local save directory when `push_to_hub` is false |

### Training

```bash
cd src
python -m finetune.finetune_t3
```

By default, the script loads `finetune/configs/finetune_turbo.yaml`. To use a different config:

```bash
python -m finetune.finetune_t3 --config finetune/configs/finetune_mtl.yaml
```

**What happens during training:**

1. The Chatterbox model is loaded (from Hub or local directory)
2. Only the T3 language model is trained; the voice encoder and S3Gen decoder remain frozen
3. Trainer checkpoints are saved to `output_dir/` at the configured interval
4. After training completes, a `final_model/` directory is created containing:
   - The finetuned T3 weights (saved with the variant-appropriate filename)
   - All other model files copied from the original model directory

The `final_model/` directory is a complete, self-contained model directory that can be loaded directly with `ChatterboxTTS.from_local()` (or the corresponding class for your variant).

**Resume from checkpoint:**

Set `resume_from_checkpoint: true` in the config, or point it to a specific checkpoint path.

### Hyperparameter Search

Run an automated hyperparameter search over learning rate, warmup steps, gradient clipping, and accumulation steps:

```bash
cd src
python -m finetune.hyperparam_search --strategy grid
python -m finetune.hyperparam_search --strategy random --n_trials 20
python -m finetune.hyperparam_search --strategy optuna --n_trials 50
```

Use `--dry_run` to test the search logic without running actual training. The search space is defined at the top of `src/finetune/hyperparam_search.py`.

### Utilities

#### Checkpoint Conversion

Convert Trainer checkpoints into standalone model directories compatible with `from_local()`:

```bash
cd src

# Convert a single checkpoint
python -m finetune.utils.convert_checkpoint \
    /path/to/checkpoint-50000 \
    /path/to/original-model-dir \
    --model_variant turbo

# Convert all checkpoints from a training run
python -m finetune.utils.convert_checkpoint \
    /path/to/checkpoint-50000 \
    /path/to/original-model-dir \
    --model_variant turbo \
    --all \
    --output_dir /path/to/converted-checkpoints
```

This copies the base model files from the original model directory and replaces the T3 weights with the finetuned checkpoint weights.

#### Checkpoint Testing

Generate test audio from a checkpoint to evaluate quality:

```bash
cd src

# Default: load pretrained model, swap in finetuned T3 weights
python -m finetune.utils.test_checkpoint \
    --model_variant turbo \
    --checkpoint_dir /path/to/checkpoint-50000

# Load a complete local model directory instead
python -m finetune.utils.test_checkpoint \
    --model_variant multilingual \
    --checkpoint_dir /path/to/final_model \
    --load_mode full_model \
    --language_id da
```

The script generates audio for each prompt in `src/finetune/utils/text_examples.txt` using every voice sample found in `--voice_dir` (defaults to `./voices`). You can also specify individual voice samples with `--voice_samples`.

#### Upload Notebooks

Two Jupyter notebooks in `src/finetune/utils/` for uploading models to Hugging Face:

- **`upload_model.ipynb`** -- Upload a complete model directory (e.g. `final_model/`) directly. Validates that all required files for the selected variant are present.
- **`upload_checkpoint.ipynb`** -- Assemble a clean model from the pretrained base model + finetuned T3 checkpoint weights, then upload. Use this when you only have a Trainer checkpoint, not a full model directory.

Both notebooks stage files into a clean temporary directory before upload to avoid including unwanted artifacts.

## Inference

`ChatterboxInference` is the recommended entry point for all inference use cases. It wraps the base, multilingual, and turbo model variants and adds text normalisation, sentence splitting, inter-sentence silence, and Hub/local model loading.

> **Intended use:** `ChatterboxInference` is designed for single-caller scenarios — local scripts, research pipelines, and single-worker servers. It is not suited for concurrent multi-user serving, where each request would require its own model instance. Production-scale serving with request batching and concurrency would require a dedicated inference server implementation.

### Loading

```python
import torchaudio
from chatterbox.inference import ChatterboxInference

# Default pretrained model from Hugging Face Hub
model = ChatterboxInference.from_pretrained(model_type="multilingual", language="da", device="cuda")

# Custom Hub repo (e.g. a finetuned model)
model = ChatterboxInference.from_pretrained(
    model_type="multilingual",
    language="da",
    device="cuda",
    repo_id="CoRal-project/roest-v3-chatterbox-500m",
)

# Local model directory
model = ChatterboxInference.from_local("path/to/model", model_type="multilingual", language="da", device="cuda")
```

`model_type` is one of `"base"`, `"multilingual"`, or `"turbo"`. When using a finetuned checkpoint, match the `model_type` to the variant the checkpoint was trained from.

### Blocking generation

```python
# Standard path — works on CPU and CUDA
wav = model.generate("Hej, verden.", audio_prompt_path="reference.wav")

# Fast CUDA graph path — ~2× faster T3 decode on GPU; falls back to generate() on CPU
wav = model.generate_fast("Hej, verden.", audio_prompt_path="reference.wav")

torchaudio.save("output.wav", wav, model.sr)
```

`generate_fast()` accepts the same parameters as `generate()`. The first call per session captures the CUDA graph (one-time cost); subsequent calls reuse it.

### Text preprocessing

Before synthesis, `ChatterboxInference` applies two preprocessing steps by default (both can be disabled per-call):

- **Text normalisation** — numbers are expanded to words in a language-aware way using [`num2words`](https://github.com/savoirfairelinux/num2words). For example, `"Der er 1.000 deltagere"` becomes `"Der er et tusinde deltagere"` in Danish, and `"There are 1,000 attendees"` becomes `"There are one thousand attendees"` in English. Thousands separators and decimal conventions are handled per language.
- **Sentence splitting** — long input is split into sentences using NLTK's `sent_tokenize`, with per-language tokenisation models for 18 languages. Each sentence is synthesised independently and concatenated with a configurable inter-sentence silence (default 100 ms). This keeps individual T3 decode sequences short and improves prosody at sentence boundaries.

Both steps are controlled via the `normalize_text` and `sentence_split` flags on the constructor or per-call.

### Streaming

Streaming methods always split on sentences — each yielded `torch.Tensor` corresponds to one sentence and is ready to play back as soon as it is synthesised, without waiting for the rest. Concatenating all chunks produces the same audio as the equivalent blocking call.

| Method | Style | Fast path |
|---|---|---|
| `generate_stream_sync` | sync generator | no |
| `generate_stream_async` | async generator | no |
| `generate_stream_fast_sync` | sync generator | yes (CUDA) |
| `generate_stream_fast_async` | async generator | yes (CUDA) |

```python
text = "Første sætning fylder lidt. Anden sætning kommer hurtigt efter. Tredje sætning afslutter det hele."

# Sync — each chunk is one sentence, ready to play as soon as it is synthesised
for chunk in model.generate_stream_fast_sync(text, audio_prompt_path="ref.wav"):
    send_to_playback(chunk)

# Async — same behaviour inside a FastAPI or other async server
async for chunk in model.generate_stream_fast_async(text, audio_prompt_path="ref.wav"):
    await send_to_client(chunk)
```

### Speaker conditioning

`prepare_conditionals()` pre-computes and caches speaker embeddings from a reference audio file. Call it once to avoid re-encoding on every sentence when reusing the same voice across multiple calls:

```python
model.prepare_conditionals("reference.wav")
wav1 = model.generate("First sentence.")
wav2 = model.generate("Second sentence.")  # no re-encoding
```

## Repository Structure

```
src/
  chatterbox/              # Chatterbox inference code (extended with fast CUDA inference and streaming)
    inference.py           # ChatterboxInference wrapper (added)
    utils/
      normalizer.py        # Language-aware text normalisation, e.g. number-to-words (added)
      splitter.py          # Sentence splitting via NLTK (added)
      device.py            # Device utilities (added)
  finetune/                # Finetuning framework (this fork's addition)
    finetune_t3.py         # Main training entry point
    preprocess_dataset.py  # Dataset preprocessing (speaker embeddings)
    hyperparam_search.py   # Automated hyperparameter search
    custom_arguments.py    # Argument dataclasses (all config fields)
    custom_models.py       # HuggingFace Trainer-compatible T3 wrapper
    dataset.py             # Dataset and data collator
    load_data.py           # Dataset loading logic
    load_model.py          # Model loading (local / Hub / pretrained)
    configs/               # YAML configuration files
    utils/
      convert_checkpoint.py     # Checkpoint to model directory conversion
      test_checkpoint.py        # Audio generation for checkpoint evaluation
      upload_model.ipynb        # Upload complete model to HF
      upload_checkpoint.ipynb   # Upload checkpoint-assembled model to HF
      text_examples.txt         # Test prompts for checkpoint testing
```

## Acknowledgements

- [Resemble AI / Chatterbox](https://github.com/resemble-ai/chatterbox) -- the upstream TTS models
- [stlohrey/chatterbox-finetuning](https://github.com/stlohrey/chatterbox-finetuning) -- original finetuning scripts that inspired this framework
- [CoRal project](https://coral.alexandra.dk/) / [Alexandra Institute](https://www.alexandra.dk/) -- finetuning framework development


## Citation

If you use the Chatterbox models, please cite the original work:

```
@misc{roest-hatterbox,
  author    = {Daniel Christopher Biørrith, Dan Saattrup Nielsen, Sif Bernstorff Lehmann, Simon Leminen Madsen and Torben Blach},
  title     = {Røst-v3-chatterbox-500m: A Danish state-of-the-art text-to-speech model},
  year      = {2026},
  url       = {https://huggingface.co/CoRal-project/roest-v3-chatterbox-500m},
}
```

## Roadmap

- [x] Streaming inference support
- [x] Optimised generation (reduced latency, faster decoding)

