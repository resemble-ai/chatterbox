# Specifications: Apple Silicon Optimization

## 1. Introduction

This document outlines the plan to optimize the Chatterbox project for Apple Silicon (M-series) Macs. The primary goal is to enable GPU acceleration by leveraging PyTorch's Metal Performance Shaders (MPS) backend. This will provide significant performance improvements for users running this project on modern Mac hardware.

## 2. Motivation

Deep learning models for Text-to-Speech (TTS) and Voice Conversion (VC) are computationally intensive. Running them on the CPU can be slow, leading to a poor user experience. Apple Silicon chips have powerful integrated GPUs that can be used for general-purpose computing. By using the MPS backend, we can offload the model inference to the GPU, resulting in a dramatic speed-up and making the application much more responsive.

The project currently has inconsistent and incomplete support for device selection. This effort will standardize it and make MPS the default on supported hardware.

## 3. Proposed Changes

The optimization will be achieved through the following code modifications:

### 3.1. Standardized Device Selection

A consistent device selection logic will be implemented and used across the project. The logic will check for device availability in the following order of priority:

1.  **MPS** (for Apple Silicon GPUs)
2.  **CUDA** (for NVIDIA GPUs)
3.  **CPU** (as a fallback)

A helper function or a consistent code snippet will be used to determine the device at runtime. An example of this logic is already present in `example_tts.py` and will be adopted more broadly.

```python
import torch

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
```

### 3.2. Model and Tensor Placement

All PyTorch models and data tensors will be explicitly moved to the selected device using the `.to(device)` method. This ensures that all computations are performed on the accelerated hardware.

In line with the project's requirements, model weights will be loaded onto the CPU first (`map_location='cpu'`) before being moved to the target device. This is a good practice for memory management, especially with large models.

### 3.3. File-Specific Modifications

The following files have been identified for modification:

*   **`gradio_tts_app.py` & `gradio_vc_app.py`**: The current device selection logic is limited to CUDA and CPU. It will be updated to include MPS.
*   **`src/chatterbox/tts.py` & `src/chatterbox/vc.py`**: These core modules will be reviewed to ensure the device is handled correctly and consistently. The existing device handling logic will be updated to the new standard.
*   **`example_tts.py` & `example_vc.py`**: These files already contain MPS detection logic. They will be reviewed to ensure they align with the standardized approach and updated if necessary.
*   **`example_for_mac.py`**: This script will be updated with the new device selection logic.

## 4. Success Criteria

*   The project successfully utilizes the GPU on Apple Silicon Macs for model inference.
*   A noticeable performance improvement (reduced latency) is observed for TTS and VC tasks on Apple Silicon hardware.
*   The project remains fully functional on non-Mac platforms (e.g., Linux with CUDA GPUs, or systems with only CPU support).
*   The code is clean, and device selection is handled consistently across the codebase.

Once this specification is approved, I will proceed with the implementation of the changes. 