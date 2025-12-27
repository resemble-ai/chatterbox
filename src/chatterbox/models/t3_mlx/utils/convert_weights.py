# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
Weight conversion utilities for PyTorch to MLX.
Handles loading PyTorch weights and converting them to MLX format.
"""

import logging
from pathlib import Path
from typing import Dict, Union
import numpy as np
import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def convert_t3_weights_to_mlx(
    pytorch_ckpt_path: Union[str, Path], mlx_ckpt_path: Union[str, Path]
):
    """
    Convert T3 weights from PyTorch safetensors format to MLX-compatible format.

    Args:
        pytorch_ckpt_path: Path to PyTorch checkpoint (.safetensors)
        mlx_ckpt_path: Path to save MLX weights (.npz)
    """
    from safetensors.torch import load_file as load_safetensors_torch

    logger.info(f"Loading PyTorch weights from {pytorch_ckpt_path}")
    pytorch_ckpt_path = Path(pytorch_ckpt_path)
    mlx_ckpt_path = Path(mlx_ckpt_path)

    # Load PyTorch weights
    state_dict = load_safetensors_torch(pytorch_ckpt_path)

    # Convert to numpy (intermediate format)
    mlx_state_dict = {}
    total_params = 0

    for key, value in state_dict.items():
        if hasattr(value, "cpu"):
            # PyTorch tensor
            numpy_array = value.cpu().numpy()
            mlx_state_dict[key] = numpy_array
            total_params += numpy_array.size
        else:
            # Already numpy or other format
            mlx_state_dict[key] = value
            if hasattr(value, "size"):
                total_params += value.size

    # Save in NumPy .npz format (MLX can load this directly)
    logger.info(f"Saving MLX weights to {mlx_ckpt_path}")
    np.savez(mlx_ckpt_path, **mlx_state_dict)

    logger.info(f"Conversion complete! Total parameters: {total_params:,}")
    logger.info(f"Weights saved to: {mlx_ckpt_path}")

    return mlx_state_dict


def load_mlx_weights(
    model: nn.Module, ckpt_path: Union[str, Path], strict: bool = True
) -> nn.Module:
    """
    Load MLX weights into a model.

    Args:
        model: MLX model to load weights into
        ckpt_path: Path to MLX checkpoint (.npz)
        strict: Whether to strictly enforce weight matching

    Returns:
        Model with loaded weights
    """
    logger.info(f"Loading MLX weights from {ckpt_path}")
    ckpt_path = Path(ckpt_path)

    # Load numpy weights
    state_dict_np = np.load(ckpt_path)

    # Convert to MLX arrays
    mlx_weights = {}
    for key in state_dict_np.files:
        mlx_weights[key] = mx.array(state_dict_np[key])

    # Load into model
    # MLX models have update() method for loading weights
    try:
        # Get model's current parameters
        model_params = dict(model.named_parameters())

        # Match and load weights
        matched = 0
        missing = []
        unexpected = []

        for name, param in model_params.items():
            if name in mlx_weights:
                # Check shape compatibility
                if param.shape == mlx_weights[name].shape:
                    # Update parameter
                    setattr(model, name, mlx_weights[name])
                    matched += 1
                else:
                    logger.warning(
                        f"Shape mismatch for {name}: "
                        f"model={param.shape}, checkpoint={mlx_weights[name].shape}"
                    )
                    if strict:
                        raise ValueError(f"Shape mismatch for parameter {name}")
            else:
                missing.append(name)

        for name in mlx_weights.keys():
            if name not in model_params:
                unexpected.append(name)

        logger.info(f"Loaded {matched} parameters")
        if missing:
            logger.warning(
                f"Missing keys in checkpoint: {missing[:10]}{'...' if len(missing) > 10 else ''}"
            )
        if unexpected:
            logger.warning(
                f"Unexpected keys in checkpoint: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}"
            )

        if strict and (missing or unexpected):
            raise ValueError(
                f"Strict mode: {len(missing)} missing, {len(unexpected)} unexpected keys"
            )

    except Exception as e:
        logger.error(f"Error loading weights: {e}")
        raise

    logger.info("Weights loaded successfully")
    return model


def save_mlx_weights(model: nn.Module, ckpt_path: Union[str, Path]):
    """
    Save MLX model weights to file.

    Args:
        model: MLX model to save
        ckpt_path: Path to save checkpoint (.npz)
    """
    logger.info(f"Saving MLX weights to {ckpt_path}")
    ckpt_path = Path(ckpt_path)

    # Get model parameters
    model_params = dict(model.named_parameters())

    # Convert to numpy for saving
    state_dict_np = {}
    for name, param in model_params.items():
        state_dict_np[name] = np.array(param)

    # Save as .npz
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(ckpt_path, **state_dict_np)

    total_params = sum(p.size for p in state_dict_np.values())
    logger.info(
        f"Saved {len(state_dict_np)} parameters ({total_params:,} total) to {ckpt_path}"
    )


def pytorch_to_mlx_tensor(tensor) -> mx.array:
    """
    Convert a PyTorch tensor to MLX array.

    Args:
        tensor: PyTorch tensor

    Returns:
        MLX array
    """
    if hasattr(tensor, "cpu"):
        return mx.array(tensor.cpu().numpy())
    elif isinstance(tensor, np.ndarray):
        return mx.array(tensor)
    elif isinstance(tensor, mx.array):
        return tensor
    else:
        raise TypeError(f"Cannot convert {type(tensor)} to MLX array")


def mlx_to_pytorch_tensor(array: mx.array):
    """
    Convert an MLX array to PyTorch tensor.

    Args:
        array: MLX array

    Returns:
        PyTorch tensor
    """
    import torch

    return torch.from_numpy(np.array(array))


def convert_state_dict_to_mlx(pytorch_state_dict: Dict) -> Dict[str, mx.array]:
    """
    Convert entire PyTorch state dict to MLX format.

    Args:
        pytorch_state_dict: Dictionary of PyTorch tensors

    Returns:
        Dictionary of MLX arrays
    """
    mlx_state_dict = {}
    for key, value in pytorch_state_dict.items():
        mlx_state_dict[key] = pytorch_to_mlx_tensor(value)
    return mlx_state_dict


def convert_state_dict_to_pytorch(mlx_state_dict: Dict[str, mx.array]) -> Dict:
    """
    Convert entire MLX state dict to PyTorch format.

    Args:
        mlx_state_dict: Dictionary of MLX arrays

    Returns:
        Dictionary of PyTorch tensors
    """
    pytorch_state_dict = {}
    for key, value in mlx_state_dict.items():
        pytorch_state_dict[key] = mlx_to_pytorch_tensor(value)
    return pytorch_state_dict
