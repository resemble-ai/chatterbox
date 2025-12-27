#!/usr/bin/env python3
# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
Weight Conversion Utility for S3Gen MLX

Converts PyTorch S3Gen checkpoint weights to MLX format.
Handles:
1. Weight name mapping (PyTorch nn.ModuleList -> MLX naming)
2. Conv1d weight transpose (PyTorch [O,I,K] -> MLX [O,K,I])
3. Weight normalization (weight_g + weight_v -> weight)
4. Transformer attention renaming

Usage:
    python convert_weights.py --input path/to/s3gen.safetensors --output path/to/s3gen_mlx.npz

Or programmatically:
    from chatterbox.models.s3gen_mlx.convert_weights import convert_and_load_weights
    mlx_weights = convert_and_load_weights(pytorch_checkpoint_path)
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Weight Format Conversion Functions
# ============================================================================


def convert_conv1d_weight(weight: np.ndarray) -> np.ndarray:
    """
    Convert Conv1d weight from PyTorch to MLX format.

    PyTorch: [out_channels, in_channels, kernel_size]
    MLX:     [out_channels, kernel_size, in_channels]
    """
    if len(weight.shape) != 3:
        return weight
    return np.transpose(weight, (0, 2, 1))


def convert_conv_transpose1d_weight(weight: np.ndarray) -> np.ndarray:
    """
    Convert ConvTranspose1d weight from PyTorch to MLX format.

    PyTorch: [in_channels, out_channels, kernel_size]
    MLX:     [out_channels, kernel_size, in_channels]
    """
    if len(weight.shape) != 3:
        return weight
    # [in, out, kernel] -> [out, kernel, in]
    return np.transpose(weight, (1, 2, 0))


def convert_conv2d_weight(weight: np.ndarray) -> np.ndarray:
    """
    Convert Conv2d weight from PyTorch to MLX format.

    PyTorch: [out_channels, in_channels, H, W]
    MLX:     [out_channels, H, W, in_channels]
    """
    if len(weight.shape) != 4:
        return weight
    return np.transpose(weight, (0, 2, 3, 1))


def convert_weight_norm(weight_g: np.ndarray, weight_v: np.ndarray) -> np.ndarray:
    """
    Combine weight_norm parameters into a single weight.

    PyTorch's weight_norm stores:
    - weight_g: scalar per output channel [out_channels, 1, ...]
    - weight_v: unnormalized weight [out_channels, in_channels, ...]

    Combined: weight = weight_g * (weight_v / ||weight_v||)
    """
    # Reshape weight_g for broadcasting
    if len(weight_v.shape) == 3:  # Conv1d
        norm = np.linalg.norm(
            weight_v.reshape(weight_v.shape[0], -1), axis=1, keepdims=True
        )
        norm = norm.reshape(weight_v.shape[0], 1, 1)
    elif len(weight_v.shape) == 2:  # Linear
        norm = np.linalg.norm(weight_v, axis=1, keepdims=True)
    else:
        norm = np.linalg.norm(
            weight_v.reshape(weight_v.shape[0], -1), axis=1, keepdims=True
        )
        norm = norm.reshape((weight_v.shape[0],) + (1,) * (len(weight_v.shape) - 1))

    return weight_g * (weight_v / (norm + 1e-12))


# ============================================================================
# Name Mapping Functions
# ============================================================================


def map_decoder_block_name(key: str) -> str:
    """
    Map decoder block names from PyTorch to MLX.

    PyTorch structure (nn.ModuleList):
        down_blocks.0.0 = resnet
        down_blocks.0.1.0 = first transformer_block
        down_blocks.0.2 = downsample

    MLX structure (explicit attributes):
        down_blocks_0.layers_0 = resnet
        down_blocks_0.layers_1_0 = first transformer_block
        down_blocks_0.layers_2 = downsample
    """
    patterns = [
        # down_blocks.X.0.xxx -> down_blocks_X.layers_0.xxx (resnet)
        (r"down_blocks\.(\d+)\.0\.", r"down_blocks_\1.layers_0."),
        # down_blocks.X.1.Y.xxx -> down_blocks_X.layers_1_Y.xxx (transformer blocks)
        (r"down_blocks\.(\d+)\.1\.(\d+)\.", r"down_blocks_\1.layers_1_\2."),
        # down_blocks.X.2.xxx -> down_blocks_X.layers_2.xxx (downsample)
        (r"down_blocks\.(\d+)\.2\.", r"down_blocks_\1.layers_2."),
        # Same for mid_blocks (only has layers_0 and layers_1)
        (r"mid_blocks\.(\d+)\.0\.", r"mid_blocks_\1.layers_0."),
        (r"mid_blocks\.(\d+)\.1\.(\d+)\.", r"mid_blocks_\1.layers_1_\2."),
        # Same for up_blocks
        (r"up_blocks\.(\d+)\.0\.", r"up_blocks_\1.layers_0."),
        (r"up_blocks\.(\d+)\.1\.(\d+)\.", r"up_blocks_\1.layers_1_\2."),
        (r"up_blocks\.(\d+)\.2\.", r"up_blocks_\1.layers_2."),
    ]

    result = key
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)
    return result


def map_resnet_block_name(key: str) -> str:
    """
    Map ResNet block names from PyTorch to MLX.

    PyTorch: block1.block.0.weight (Sequential with CausalConv1d at index 0)
    MLX:     block1.conv.weight (CausalBlock1DMLX has conv attribute)

    PyTorch: block1.block.2.weight (LayerNorm at index 2)
    MLX:     block1.norm.weight

    PyTorch: mlp.1.weight (nn.Sequential index)
    MLX:     mlp_1.weight (explicit attribute)
    """
    patterns = [
        # block1.block.0 -> block1.conv (first is conv)
        (r"\.block1\.block\.0\.", ".block1.conv."),
        # block1.block.2 -> block1.norm (third is norm after transpose)
        (r"\.block1\.block\.2\.", ".block1.norm."),
        # block2.block.0 -> block2.conv
        (r"\.block2\.block\.0\.", ".block2.conv."),
        # block2.block.2 -> block2.norm
        (r"\.block2\.block\.2\.", ".block2.norm."),
        # mlp.1 -> mlp_1 (explicit attribute in ResnetBlock1DMLX)
        (r"\.mlp\.1\.", ".mlp_1."),
    ]

    result = key
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)
    return result


def map_causal_conv_name(key: str) -> str:
    """
    Map causal conv names from PyTorch to MLX.

    For downsample/upsample that use CausalConv1d directly:
    PyTorch: down_blocks.0.2.weight (CausalConv1d is the module directly)
    MLX:     down_blocks.0.layers_2.conv.weight (CausalConv1DMLX wraps conv)

    For final block:
    PyTorch: final_block.block.0.weight
    MLX:     final_block.conv.weight
    """
    patterns = [
        # layers_2.weight -> layers_2.conv.weight (for CausalConv1DMLX downsample)
        # Only if it's directly at layers_2 level without further nesting
        (r"(layers_2)\.(weight|bias)$", r"\1.conv.\2"),
        # final_block.block.0 -> final_block.conv
        (r"final_block\.block\.0\.", "final_block.conv."),
        # final_block.block.2 -> final_block.norm
        (r"final_block\.block\.2\.", "final_block.norm."),
    ]

    result = key
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)
    return result


def map_transformer_block_name(key: str) -> str:
    """
    Map transformer block names from PyTorch to MLX.

    PyTorch: attn1.to_out.0.weight (nn.Linear in Sequential)
    MLX:     attn1.to_out.weight (direct nn.Linear)

    PyTorch: ff.net.0.proj.weight (GEGLU first projection)
    MLX:     ff.proj_in.weight

    PyTorch: ff.net.2.weight (output projection)
    MLX:     ff.proj_out.weight
    """
    patterns = [
        # attn1.to_out.0 -> attn1.to_out (remove Sequential index)
        (r"\.attn1\.to_out\.0\.", ".attn1.to_out."),
        (r"\.attn2\.to_out\.0\.", ".attn2.to_out."),
        # ff.net.0.proj -> ff.proj_in (GEGLU)
        (r"\.ff\.net\.0\.proj\.", ".ff.proj_in."),
        # ff.net.2 -> ff.proj_out
        (r"\.ff\.net\.2\.", ".ff.proj_out."),
    ]

    result = key
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)
    return result


def map_time_mlp_name(key: str) -> str:
    """
    Map time MLP names from PyTorch to MLX.

    PyTorch: mlp.1.weight (nn.Sequential with Linear at index 1)
    MLX:     mlp.layers.1.weight (MLX Sequential uses 'layers' prefix)
    """
    # Actually, looking at the MLX output, it seems mlp uses layers.1 automatically
    # from nn.Sequential. Let's verify the actual MLX structure first.
    # For now, we might not need this mapping.
    return key


def map_encoder_name(key: str) -> str:
    """
    Map conformer encoder names from PyTorch to MLX.

    PyTorch: encoder.embed.out.0.weight (Linear in Sequential)
    MLX:     encoder.embed.linear.weight

    PyTorch: encoder.embed.out.1.weight (LayerNorm in Sequential)
    MLX:     encoder.embed.norm.weight

    PyTorch: encoder.encoders.0.xxx (ModuleList)
    MLX:     encoder.encoders_0.xxx (explicit attributes)

    Same for up_embed and up_encoders in upsample encoder.
    """
    patterns = [
        # embed.out.0 -> embed.linear
        (r"\.embed\.out\.0\.", ".embed.linear."),
        # embed.out.1 -> embed.norm
        (r"\.embed\.out\.1\.", ".embed.norm."),
        # up_embed.out.0 -> up_embed.linear
        (r"\.up_embed\.out\.0\.", ".up_embed.linear."),
        # up_embed.out.1 -> up_embed.norm
        (r"\.up_embed\.out\.1\.", ".up_embed.norm."),
        # encoders.X -> encoders_X (explicit attributes)
        (r"\.encoders\.(\d+)\.", r".encoders_\1."),
        # up_encoders.X -> up_encoders_X (explicit attributes)
        (r"\.up_encoders\.(\d+)\.", r".up_encoders_\1."),
    ]

    result = key
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)
    return result


def map_hifigan_name(key: str) -> str:
    """
    Map HiFiGAN vocoder names from PyTorch to MLX.

    Handles:
    - F0 predictor condnet Sequential -> conv1, conv2, etc.
    - ups.X -> ups_X
    - resblocks.X -> resblocks_X
    - source_downs.X -> source_downs_X
    - source_resblocks.X -> source_resblocks_X
    - resblocks.X.convs1.Y -> resblocks_X.convs1_Y
    - resblocks.X.activations1.Y -> resblocks_X.activations1_Y
    - source_resblocks.X.convs1.Y -> source_resblocks_X.convs1_Y
    - convs1.X -> convs1_X (for ResBlock internal layers)
    """
    patterns = [
        # f0_predictor.condnet.0 -> f0_predictor.conv1
        (r"\.f0_predictor\.condnet\.0\.", ".f0_predictor.conv1."),
        (r"\.f0_predictor\.condnet\.2\.", ".f0_predictor.conv2."),
        (r"\.f0_predictor\.condnet\.4\.", ".f0_predictor.conv3."),
        (r"\.f0_predictor\.condnet\.6\.", ".f0_predictor.conv4."),
        (r"\.f0_predictor\.condnet\.8\.", ".f0_predictor.conv5."),
        # ups.X -> ups_X (with or without trailing dot/end)
        (r"\.ups\.(\d+)(\.|\Z)", r".ups_\1\2"),
        # source_downs.X -> source_downs_X (with or without trailing dot/end)
        (r"\.source_downs\.(\d+)(\.|\Z)", r".source_downs_\1\2"),
        # source_resblocks.X.convs1.Y -> source_resblocks_X.convs1_Y (must come before source_resblocks_X)
        (
            r"\.source_resblocks\.(\d+)\.convs1\.(\d+)(\.|\Z)",
            r".source_resblocks_\1.convs1_\2\3",
        ),
        (
            r"\.source_resblocks\.(\d+)\.convs2\.(\d+)(\.|\Z)",
            r".source_resblocks_\1.convs2_\2\3",
        ),
        (
            r"\.source_resblocks\.(\d+)\.activations1\.(\d+)(\.|\Z)",
            r".source_resblocks_\1.activations1_\2\3",
        ),
        (
            r"\.source_resblocks\.(\d+)\.activations2\.(\d+)(\.|\Z)",
            r".source_resblocks_\1.activations2_\2\3",
        ),
        # source_resblocks.X (if not already matched) -> source_resblocks_X
        (r"\.source_resblocks\.(\d+)(\.|\Z)", r".source_resblocks_\1\2"),
        # resblocks.X.convs1.Y -> resblocks_X.convs1_Y
        (r"\.resblocks\.(\d+)\.convs1\.(\d+)(\.|\Z)", r".resblocks_\1.convs1_\2\3"),
        (r"\.resblocks\.(\d+)\.convs2\.(\d+)(\.|\Z)", r".resblocks_\1.convs2_\2\3"),
        (
            r"\.resblocks\.(\d+)\.activations1\.(\d+)(\.|\Z)",
            r".resblocks_\1.activations1_\2\3",
        ),
        (
            r"\.resblocks\.(\d+)\.activations2\.(\d+)(\.|\Z)",
            r".resblocks_\1.activations2_\2\3",
        ),
        # resblocks.X (if not already matched) -> resblocks_X
        (r"\.resblocks\.(\d+)(\.|\Z)", r".resblocks_\1\2"),
        # For standalone HiFiGAN: convs1.X -> convs1_X (at end of key or before .)
        (r"\.convs1\.(\d+)(\.|\Z)", r".convs1_\1\2"),
        (r"\.convs2\.(\d+)(\.|\Z)", r".convs2_\1\2"),
        (r"\.activations1\.(\d+)(\.|\Z)", r".activations1_\1\2"),
        (r"\.activations2\.(\d+)(\.|\Z)", r".activations2_\1\2"),
    ]

    result = key
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)
    return result


def map_speaker_encoder_name(key: str) -> str:
    """
    Map speaker encoder (CAMPPlus) names from PyTorch to MLX.

    PyTorch xvector structure:
        speaker_encoder.xvector.block1.tdnnd1.*
        speaker_encoder.xvector.block1.tdnnd10.*  (up to 12 layers in block1)
        speaker_encoder.xvector.block2.tdnnd24.*  (up to 24 layers in block2)
        speaker_encoder.xvector.block3.tdnnd16.*  (up to 16 layers in block3)

    MLX xvector structure (same naming, xvector becomes top-level in CAMPPlusMLX):
        speaker_encoder.block1.tdnnd1.*
        speaker_encoder.block1.tdnnd10.*
        etc.

    PyTorch head structure:
        speaker_encoder.head.layer1.0.shortcut.0.weight  (conv)
        speaker_encoder.head.layer1.0.shortcut.1.bias    (bn)

    MLX head structure:
        speaker_encoder.head.layer1_0.shortcut_conv.weight
        speaker_encoder.head.layer1_0.shortcut_bn.bias
    """
    result = key

    # xvector.blockX.tdnndY -> blockX.tdnndY (remove xvector prefix)
    result = re.sub(r"\.xvector\.block(\d+)\.", r".block\1.", result)

    # xvector.transit1 -> transit1 (transit layers)
    result = re.sub(r"\.xvector\.transit(\d+)\.", r".transit\1.", result)

    # xvector.tdnn -> tdnn (initial TDNN)
    result = re.sub(r"\.xvector\.tdnn\.", r".tdnn.", result)

    # xvector.dense -> dense
    result = re.sub(r"\.xvector\.dense\.", r".dense.", result)

    # xvector.out_nonlinear.batchnorm -> out_bn
    result = re.sub(r"\.xvector\.out_nonlinear\.batchnorm\.", r".out_bn.", result)
    # xvector.out_bn -> out_bn
    result = re.sub(r"\.xvector\.out_bn\.", r".out_bn.", result)

    # head.layer1.0 -> head.layer1_0, head.layer2.0 -> head.layer2_0
    result = re.sub(r"\.head\.layer(\d+)\.(\d+)\.", r".head.layer\1_\2.", result)

    # shortcut.0 -> shortcut_conv, shortcut.1 -> shortcut_bn
    result = re.sub(r"\.shortcut\.0\.", r".shortcut_conv.", result)
    result = re.sub(r"\.shortcut\.1\.", r".shortcut_bn.", result)

    # nonlinear1.batchnorm -> bn1, nonlinear2.batchnorm -> bn2 (CAMDenseTDNNLayer)
    result = re.sub(r"\.nonlinear1\.batchnorm\.", r".bn1.", result)
    result = re.sub(r"\.nonlinear2\.batchnorm\.", r".bn2.", result)

    # nonlinear1.linear -> linear1 (CAMDenseTDNNLayer)
    result = re.sub(r"\.nonlinear1\.linear\.", r".linear1.", result)

    # nonlinear.batchnorm -> bn (TransitLayer, TDNNLayer, DenseLayer)
    result = re.sub(r"\.nonlinear\.batchnorm\.", r".bn.", result)

    # .linear.weight in TDNN/Transit -> .conv.weight (they use Conv1d)
    # Note: TDNNLayer uses .conv, TransitLayer uses .linear
    result = re.sub(r"\.tdnn\.linear\.", r".tdnn.conv.", result)

    # transit.bn.linear -> transit.linear (TransitLayer)
    # Actually: TransitLayer has bn and linear separately
    # Pattern: transit1.nonlinear.linear -> transit1.linear
    # But wait - TransitLayer uses Conv1d named 'linear', not 'conv'

    return result


def map_full_name(key: str) -> str:
    """
    Apply all name mappings to convert PyTorch key to MLX key.
    """
    result = key

    # Apply mappings in order (order matters!)
    result = map_decoder_block_name(result)
    result = map_resnet_block_name(result)
    result = map_causal_conv_name(result)  # Must come after decoder_block_name
    result = map_transformer_block_name(result)
    result = map_encoder_name(result)  # Conformer encoder mapping
    result = map_speaker_encoder_name(result)  # CAMPPlus speaker encoder
    result = map_time_mlp_name(result)
    result = map_hifigan_name(result)  # HiFiGAN and F0 predictor

    return result


# ============================================================================
# Main Conversion Functions
# ============================================================================


def convert_s3gen_state_dict(
    pytorch_state_dict: Dict[str, Any], verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Convert entire S3Gen state dict from PyTorch to MLX format.

    Args:
        pytorch_state_dict: PyTorch state dictionary
        verbose: Print detailed conversion info

    Returns:
        MLX-compatible state dictionary with numpy arrays
    """
    import torch

    mlx_state = {}
    weight_norm_buffers = {}  # Collect weight_norm params (old and new format)
    converted_count = 0
    skipped_count = 0

    # First pass: collect weight_norm parameters
    # Handles both old (weight_g/weight_v) and new (parametrizations.weight.original0/1) formats
    for key, value in pytorch_state_dict.items():
        if hasattr(value, "numpy"):
            np_value = value.numpy()
        elif isinstance(value, torch.Tensor):
            np_value = value.detach().cpu().numpy()
        else:
            np_value = np.array(value)

        # Old weight_norm format: weight_g, weight_v
        if "weight_g" in key:
            base_key = key.replace(".weight_g", "")
            if base_key not in weight_norm_buffers:
                weight_norm_buffers[base_key] = {}
            weight_norm_buffers[base_key]["g"] = np_value
        elif "weight_v" in key:
            base_key = key.replace(".weight_v", "")
            if base_key not in weight_norm_buffers:
                weight_norm_buffers[base_key] = {}
            weight_norm_buffers[base_key]["v"] = np_value
        # New parametrizations format: parametrizations.weight.original0 (g), original1 (v)
        elif "parametrizations.weight.original0" in key:
            base_key = key.replace(".parametrizations.weight.original0", "")
            if base_key not in weight_norm_buffers:
                weight_norm_buffers[base_key] = {}
            weight_norm_buffers[base_key]["g"] = np_value
        elif "parametrizations.weight.original1" in key:
            base_key = key.replace(".parametrizations.weight.original1", "")
            if base_key not in weight_norm_buffers:
                weight_norm_buffers[base_key] = {}
            weight_norm_buffers[base_key]["v"] = np_value

    # Second pass: convert regular weights
    for key, value in pytorch_state_dict.items():
        if hasattr(value, "numpy"):
            np_value = value.numpy()
        elif isinstance(value, torch.Tensor):
            np_value = value.detach().cpu().numpy()
        else:
            np_value = np.array(value)

        # Skip weight_norm components (we'll combine them later)
        if "weight_g" in key or "weight_v" in key:
            skipped_count += 1
            continue
        if "parametrizations.weight.original" in key:
            skipped_count += 1
            continue

        # Skip num_batches_tracked (not needed for inference)
        if "num_batches_tracked" in key:
            skipped_count += 1
            continue

        # Map name from PyTorch to MLX
        new_key = map_full_name(key)

        # Handle running_mean and running_var for BatchNorm
        # MLX BatchNorm stores them directly (no prefix change needed)
        # Just need to ensure they're included

        # Check if this is a ConvTranspose weight (ups.X layers in HiFiGAN)
        is_conv_transpose = ".ups." in key or ".ups_" in key

        # Convert weight format based on shape
        if len(np_value.shape) == 3 and "weight" in key:
            if is_conv_transpose:
                # ConvTranspose1d weight
                new_value = convert_conv_transpose1d_weight(np_value)
                if verbose:
                    logger.debug(
                        f"ConvTranspose1d transpose: {key} {np_value.shape} -> {new_key} {new_value.shape}"
                    )
            else:
                # Conv1d weight
                new_value = convert_conv1d_weight(np_value)
                if verbose:
                    logger.debug(
                        f"Conv1d transpose: {key} {np_value.shape} -> {new_key} {new_value.shape}"
                    )
        elif len(np_value.shape) == 4 and "weight" in key:
            # Conv2d weight
            new_value = convert_conv2d_weight(np_value)
            if verbose:
                logger.debug(
                    f"Conv2d transpose: {key} {np_value.shape} -> {new_key} {new_value.shape}"
                )
        else:
            new_value = np_value

        mlx_state[new_key] = new_value
        converted_count += 1

        if verbose and key != new_key:
            logger.debug(f"Renamed: {key} -> {new_key}")

    # Third pass: combine weight_norm parameters
    for base_key, params in weight_norm_buffers.items():
        if "g" in params and "v" in params:
            combined = convert_weight_norm(params["g"], params["v"])

            # Check if this is a ConvTranspose weight
            is_conv_transpose = ".ups." in base_key or ".ups_" in base_key

            # Apply Conv conversion if needed
            if len(combined.shape) == 3:
                if is_conv_transpose:
                    combined = convert_conv_transpose1d_weight(combined)
                else:
                    combined = convert_conv1d_weight(combined)

            # Map the name
            new_key = map_full_name(f"{base_key}.weight")
            mlx_state[new_key] = combined
            converted_count += 1

            if verbose:
                logger.debug(f"Weight norm combined: {base_key} -> {new_key}")

    logger.info(f"Converted {converted_count} parameters, skipped {skipped_count}")
    return mlx_state


def load_pytorch_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load PyTorch checkpoint (supports .pt, .pth, .safetensors).

    Args:
        checkpoint_path: Path to PyTorch checkpoint

    Returns:
        State dictionary
    """
    path = Path(checkpoint_path)

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    else:
        import torch

        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        elif "model" in checkpoint:
            return checkpoint["model"]
        else:
            return checkpoint


def save_mlx_weights(state_dict: Dict[str, np.ndarray], output_path: str):
    """
    Save MLX weights to file.

    Args:
        state_dict: Dictionary of numpy arrays
        output_path: Output file path (supports .npz and .safetensors)
    """
    path = Path(output_path)

    if path.suffix == ".safetensors":
        from safetensors.numpy import save_file

        save_file(state_dict, str(path))
        logger.info(f"Saved MLX weights to {path}")
    else:
        np.savez(str(path), **state_dict)
        logger.info(f"Saved MLX weights to {path}")


def convert_and_load_weights(
    pytorch_checkpoint_path: str,
    output_path: Optional[str] = None,
    verbose: bool = False,
    exclude_tokenizer: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Convert PyTorch checkpoint to MLX format and optionally save.

    Args:
        pytorch_checkpoint_path: Path to PyTorch checkpoint
        output_path: Optional path to save converted weights
        verbose: Print detailed conversion info
        exclude_tokenizer: Whether to exclude tokenizer weights (not implemented in MLX)

    Returns:
        MLX-compatible state dictionary
    """
    logger.info(f"Loading PyTorch checkpoint from {pytorch_checkpoint_path}")
    pytorch_state = load_pytorch_checkpoint(pytorch_checkpoint_path)

    logger.info(f"Converting {len(pytorch_state)} parameters...")
    mlx_state = convert_s3gen_state_dict(pytorch_state, verbose=verbose)

    # Filter out tokenizer weights if requested
    if exclude_tokenizer:
        original_count = len(mlx_state)
        mlx_state = {
            k: v for k, v in mlx_state.items() if not k.startswith("tokenizer.")
        }
        filtered_count = original_count - len(mlx_state)
        if filtered_count > 0:
            logger.info(
                f"Excluded {filtered_count} tokenizer weights (not implemented in MLX)"
            )

    if output_path:
        save_mlx_weights(mlx_state, output_path)

    return mlx_state


def convert_hifigan_state_dict(
    pytorch_state_dict: Dict[str, Any], verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Convert standalone HiFiGAN PyTorch state dict to MLX format.

    This handles the HiFiGAN model weights directly without the s3gen. prefix.

    Args:
        pytorch_state_dict: PyTorch HiFiGAN state dictionary
        verbose: Print detailed conversion info

    Returns:
        MLX-compatible state dictionary with numpy arrays
    """
    import torch

    mlx_state = {}
    weight_norm_buffers = {}

    # Collect weight_norm parameters (both old and new formats)
    for key, value in pytorch_state_dict.items():
        # Old format: weight_g/weight_v
        if "weight_g" in key or "weight_v" in key:
            base_key = key.replace(".weight_g", "").replace(".weight_v", "")
            if base_key not in weight_norm_buffers:
                weight_norm_buffers[base_key] = {}
            if "weight_g" in key:
                weight_norm_buffers[base_key]["weight_g"] = (
                    value.numpy() if hasattr(value, "numpy") else value
                )
            else:
                weight_norm_buffers[base_key]["weight_v"] = (
                    value.numpy() if hasattr(value, "numpy") else value
                )

        # New format: parametrizations.weight.original0/1
        if "parametrizations.weight.original" in key:
            base_key = key.replace(".parametrizations.weight.original0", "").replace(
                ".parametrizations.weight.original1", ""
            )
            if base_key not in weight_norm_buffers:
                weight_norm_buffers[base_key] = {}
            if "original0" in key:
                weight_norm_buffers[base_key]["weight_g"] = (
                    value.numpy() if hasattr(value, "numpy") else value
                )
            else:
                weight_norm_buffers[base_key]["weight_v"] = (
                    value.numpy() if hasattr(value, "numpy") else value
                )

    # Process regular parameters
    for key, value in pytorch_state_dict.items():
        # Skip weight_norm components
        if "weight_g" in key or "weight_v" in key:
            continue
        if "parametrizations.weight.original" in key:
            continue
        if "num_batches_tracked" in key:
            continue

        # Convert to numpy
        if isinstance(value, torch.Tensor):
            value_np = value.detach().cpu().numpy()
        else:
            value_np = np.array(value)

        # Apply HiFiGAN name mapping (add prefix temporarily then remove)
        temp_key = "." + key
        mlx_key = map_hifigan_name(temp_key)[1:]  # Remove the leading dot

        # Convert weight shapes
        if "ups_" in mlx_key and ".weight" in mlx_key and len(value_np.shape) == 3:
            # ConvTranspose1d: [in, out, kernel] -> [out, kernel, in]
            value_np = convert_conv_transpose1d_weight(value_np)
        elif ".weight" in mlx_key and len(value_np.shape) == 3:
            # Regular Conv1d: [out, in, kernel] -> [out, kernel, in]
            value_np = convert_conv1d_weight(value_np)

        mlx_state[mlx_key] = value_np
        if verbose:
            print(f"  {key} -> {mlx_key} {value_np.shape}")

    # Apply weight normalization
    for base_key, buffers in weight_norm_buffers.items():
        if "weight_g" in buffers and "weight_v" in buffers:
            temp_key = "." + base_key
            mlx_key = map_hifigan_name(temp_key)[1:]

            weight = convert_weight_norm(buffers["weight_g"], buffers["weight_v"])

            # Convert weight shapes
            if "ups_" in mlx_key and len(weight.shape) == 3:
                weight = convert_conv_transpose1d_weight(weight)
            elif len(weight.shape) == 3:
                weight = convert_conv1d_weight(weight)

            mlx_state[mlx_key + ".weight"] = weight
            if verbose:
                print(f"  {base_key} (weight_norm) -> {mlx_key}.weight {weight.shape}")

    return mlx_state


def analyze_weight_mapping(pytorch_path: str, mlx_model) -> Tuple[list, list, list]:
    """
    Analyze weight mapping between PyTorch checkpoint and MLX model.

    Returns:
        Tuple of (matched_keys, unmatched_pytorch_keys, unmatched_mlx_keys)
    """
    from mlx.utils import tree_flatten

    # Load PyTorch weights
    pt_state = load_pytorch_checkpoint(pytorch_path)

    # Get MLX parameter names
    mlx_params = dict(tree_flatten(mlx_model.parameters()))
    mlx_keys = set(mlx_params.keys())

    # Convert PyTorch keys
    matched = []
    unmatched_pt = []

    for pt_key in pt_state.keys():
        # Skip weight norm components
        if "weight_g" in pt_key or "weight_v" in pt_key:
            continue
        # Note: running_mean/running_var are needed for BatchNorm inference

        mlx_key = map_full_name(pt_key)

        if mlx_key in mlx_keys:
            matched.append((pt_key, mlx_key))
            mlx_keys.discard(mlx_key)
        else:
            unmatched_pt.append((pt_key, mlx_key))

    unmatched_mlx = list(mlx_keys)

    return matched, unmatched_pt, unmatched_mlx


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Convert S3Gen PyTorch weights to MLX format"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to PyTorch checkpoint (.pt, .pth, or .safetensors)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output path for MLX weights (.npz or .safetensors)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed conversion info"
    )
    parser.add_argument(
        "--analyze",
        "-a",
        action="store_true",
        help="Analyze weight mapping without saving",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.analyze:
        # Just analyze without full model
        pt_state = load_pytorch_checkpoint(args.input)
        logger.info(f"PyTorch weights: {len(pt_state)}")

        # Show sample mappings
        logger.info("\nSample weight mappings:")
        for i, key in enumerate(sorted(pt_state.keys())[:20]):
            mlx_key = map_full_name(key)
            if key != mlx_key:
                logger.info(f"  {key}")
                logger.info(f"    -> {mlx_key}")
            else:
                logger.info(f"  {key} (unchanged)")
    else:
        mlx_state = convert_and_load_weights(
            args.input, args.output, verbose=args.verbose
        )

        logger.info("Conversion complete!")
        logger.info(f"  Input: {args.input}")
        logger.info(f"  Output: {args.output}")
        logger.info(f"  Parameters: {len(mlx_state)}")


if __name__ == "__main__":
    main()
