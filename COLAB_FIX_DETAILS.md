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

## How to Install in Colab

In a Google Colab notebook cell, running the following should now work without errors:

```python
!git clone https://github.com/resemble-ai/chatterbox.git
%cd chatterbox
!pip install -e .
```
