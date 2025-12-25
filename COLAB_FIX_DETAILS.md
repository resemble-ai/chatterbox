# Google Colab Compatibility Fixes

This document details the changes made to the `chatterbox` project to ensure compatibility with the Google Colab environment.

## Overview of Changes

The primary issue preventing installation on Google Colab was strict version pinning in `pyproject.toml`. Google Colab environments come with pre-installed versions of major libraries (like PyTorch, NumPy, Transformers) that are updated frequently. Strict pinning (e.g., `==2.6.0`) causes conflicts with these pre-installed versions or forces unnecessary and time-consuming reinstallations that may break the environment.

## File: `pyproject.toml`

The following dependencies were modified:

| Package | Original Version | New Version | Reason |
| :--- | :--- | :--- | :--- |
| `numpy` | `>=1.24.0,<1.26.0` | `>=1.26.0` | Colab often uses newer NumPy versions. Relaxed upper bound constraint. |
| `librosa` | `==0.11.0` | `>=0.10.0` | Relaxed strict pin to allow compatible newer or slightly older versions. |
| `torch` | `==2.6.0` | `>=2.0.0` | **CRITICAL**: Colab has pre-installed PyTorch. Strict pinning forces a reinstall that can break CUDA compatibility or time out. Relaxed to any major 2.x version. |
| `torchaudio` | `==2.6.0` | `>=2.0.0` | Matched `torch` relaxtion. |
| `transformers` | `==4.46.3` | `>=4.46.0` | Relaxed strict pin. Colab often has recent transformers; exact match is unnecessary. |
| `diffusers` | `==0.29.0` | `>=0.29.0` | Relaxed strict pin to allow updates. |
| `resemble-perth` | `==1.0.1` | `>=1.0.1` | Relaxed pin. |
| `conformer` | `==0.3.2` | `>=0.3.2` | Relaxed pin. |
| `safetensors` | `==0.5.3` | `>=0.5.0` | Relaxed pin. |
| `pykakasi` | `==2.3.0` | `>=2.3.0` | Relaxed pin. |
| `gradio` | `==5.44.1` | `>=4.0.0` | Relaxed largely. Gradio 5.x is new, but 4.x is often sufficient. Allowing `>=4.0.0` gives maximum flexibility. |

## File: `src/chatterbox/mtl_tts.py`

**Issue:** The project uses `torch.load` to load model checkpoints (`ve.pt`, `s3gen.pt`). These checkpoints were saved on a CUDA device.
**Fix:** Added `map_location=torch.device('cpu')` logic when the current device is CPU or MPS. This prevents `RuntimeError: Attempting to deserialize object on a CUDA device...` when running on CPU-only Colab instances.

```python
# Added to from_local method:
if device in ["cpu", "mps"]:
    map_location = torch.device('cpu')
else:
    map_location = None

# Applied 'map_location=map_location' to torch.load calls
```

## File: `src/chatterbox/tts_turbo.py`

**Issue:** `snapshot_download` was forcing `token=True`, causing `LocalTokenNotFoundError` for users without a configured Hugging Face token.
**Fix:** Changed to `token=os.getenv("HF_TOKEN")` to make authentication optional for public models.

## File: `example_tts.py`

**Issue:** The script crashed with `FileNotFoundError` if the optional `YOUR_FILE.wav` audio prompt didn't exist.
**Fix:** Added an existence check `if os.path.exists(AUDIO_PROMPT_PATH):` to skip the voice cloning example gracefully if the file is missing.



## How to Install in Colab

In a Google Colab notebook cell, running the following should now work without errors:

```python
!git clone https://github.com/resemble-ai/chatterbox.git
%cd chatterbox
!pip install -e .
```
