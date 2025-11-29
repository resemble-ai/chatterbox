# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
Quantization utilities for T3 MLX models.
Supports 4-bit and 8-bit quantization for efficient inference on Apple Silicon.
"""

import logging
from typing import Optional, Tuple
import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class QuantizedLinear(nn.Module):
    """
    Quantized linear layer for MLX.
    Uses group-wise quantization for weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 64,
        bias: bool = True,
    ):
        """
        Initialize quantized linear layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            bits: Quantization bits (4 or 8)
            group_size: Group size for quantization
            bias: Whether to use bias
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        # Quantized weights (to be set during quantization)
        self.weight_q = None  # Quantized weights
        self.scales = None    # Scaling factors
        self.zeros = None     # Zero points

        # Bias
        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None

    def quantize_weights(self, weight: mx.array):
        """
        Quantize weights using group-wise quantization.

        Args:
            weight: Original weights of shape (out_features, in_features)
        """
        out_features, in_features = weight.shape

        # Calculate number of groups
        num_groups = (in_features + self.group_size - 1) // self.group_size

        # Reshape for group-wise quantization
        # Pad if necessary
        if in_features % self.group_size != 0:
            pad_size = self.group_size - (in_features % self.group_size)
            weight = mx.pad(weight, ((0, 0), (0, pad_size)))

        weight_grouped = mx.reshape(weight, (out_features, num_groups, self.group_size))

        # Compute scales and zero points per group
        q_max = (2 ** self.bits) - 1

        # Per-group min and max
        w_min = mx.min(weight_grouped, axis=2, keepdims=True)
        w_max = mx.max(weight_grouped, axis=2, keepdims=True)

        # Compute scales and zeros
        scales = (w_max - w_min) / q_max
        zeros = -w_min / scales

        # Quantize
        weight_q = mx.round((weight_grouped - w_min) / scales)
        weight_q = mx.clip(weight_q, 0, q_max)

        # Store quantized weights and parameters
        if self.bits == 4:
            # Pack two 4-bit values into one uint8
            weight_q = weight_q.astype(mx.uint8)
        else:
            weight_q = weight_q.astype(mx.uint8)

        self.weight_q = weight_q
        self.scales = mx.squeeze(scales, axis=2)
        self.zeros = mx.squeeze(zeros, axis=2)

        logger.info(f"Quantized linear layer: {out_features}x{in_features} to {self.bits}-bit")

    def dequantize_weights(self) -> mx.array:
        """
        Dequantize weights back to float for computation.

        Returns:
            Dequantized weights
        """
        if self.weight_q is None:
            raise ValueError("Weights not quantized yet")

        # Dequantize
        weight_f = self.weight_q.astype(mx.float32)

        # Apply scales and zeros
        scales_expanded = mx.expand_dims(self.scales, 2)
        zeros_expanded = mx.expand_dims(self.zeros, 2)

        weight_deq = weight_f * scales_expanded + zeros_expanded * scales_expanded

        # Reshape back
        out_features, num_groups, group_size = weight_deq.shape
        weight_deq = mx.reshape(weight_deq, (out_features, num_groups * group_size))

        # Remove padding if necessary
        if num_groups * group_size > self.in_features:
            weight_deq = weight_deq[:, :self.in_features]

        return weight_deq

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass with quantized weights.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # Dequantize weights on-the-fly
        weight = self.dequantize_weights()

        # Standard linear operation
        output = x @ weight.T

        if self.bias is not None:
            output = output + self.bias

        return output

    @classmethod
    def from_linear(cls, linear: nn.Linear, bits: int = 4, group_size: int = 64):
        """
        Create quantized linear from standard linear layer.

        Args:
            linear: Standard nn.Linear layer
            bits: Quantization bits
            group_size: Group size

        Returns:
            QuantizedLinear layer
        """
        has_bias = linear.bias is not None

        q_linear = cls(
            in_features=linear.weight.shape[1],
            out_features=linear.weight.shape[0],
            bits=bits,
            group_size=group_size,
            bias=has_bias,
        )

        # Quantize weights
        q_linear.quantize_weights(linear.weight)

        # Copy bias
        if has_bias:
            q_linear.bias = linear.bias

        return q_linear


def quantize_model(
    model: nn.Module,
    bits: int = 4,
    group_size: int = 64,
    exclude_layers: Optional[list] = None,
) -> nn.Module:
    """
    Quantize all linear layers in a model.

    Args:
        model: Model to quantize
        bits: Quantization bits (4 or 8)
        group_size: Group size for quantization
        exclude_layers: List of layer names to exclude from quantization

    Returns:
        Quantized model
    """
    if exclude_layers is None:
        exclude_layers = []

    logger.info(f"Quantizing model to {bits}-bit with group_size={group_size}")

    # Recursively quantize linear layers
    def quantize_module(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # Check if should exclude
            if any(excl in full_name for excl in exclude_layers):
                logger.info(f"Skipping quantization for {full_name}")
                continue

            # Quantize if Linear layer
            if isinstance(child, nn.Linear):
                logger.debug(f"Quantizing {full_name}")
                quantized = QuantizedLinear.from_linear(child, bits=bits, group_size=group_size)
                setattr(module, name, quantized)
            else:
                # Recursively process child modules
                quantize_module(child, prefix=full_name)

    quantize_module(model)

    logger.info("Model quantization complete")
    return model


class QuantizedT3MLX:
    """
    Wrapper for quantized T3 MLX model with easy loading interface.
    """

    def __init__(self, model, bits: int = 4, group_size: int = 64):
        """
        Initialize quantized T3 wrapper.

        Args:
            model: T3MLX model to quantize
            bits: Quantization bits
            group_size: Group size
        """
        self.bits = bits
        self.group_size = group_size

        # Exclude embedding and output layers from quantization
        # These are sensitive and better kept in full precision
        exclude_layers = [
            'text_emb',
            'speech_emb',
            'text_head',
            'speech_head',
        ]

        self.model = quantize_model(
            model,
            bits=bits,
            group_size=group_size,
            exclude_layers=exclude_layers
        )

    def __call__(self, *args, **kwargs):
        """Forward pass through quantized model."""
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Generate with quantized model."""
        return self.model.generate(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, ckpt_path, bits: int = 4, group_size: int = 64):
        """
        Load pre-trained T3 model and quantize.

        Args:
            ckpt_path: Path to checkpoint
            bits: Quantization bits
            group_size: Group size

        Returns:
            QuantizedT3MLX instance
        """
        from ..t3_mlx import T3MLX
        from ..utils.convert_weights import load_mlx_weights

        # Load full precision model
        model = T3MLX()
        load_mlx_weights(model, ckpt_path)

        # Quantize
        return cls(model, bits=bits, group_size=group_size)


def benchmark_quantization(model, bits_list=[4, 8, 16]) -> dict:
    """
    Benchmark different quantization settings.

    Args:
        model: Model to benchmark
        bits_list: List of bit widths to test

    Returns:
        Dictionary of benchmark results
    """
    import time

    results = {}

    # Original model size
    def model_size_mb(m):
        """Estimate model size in MB."""
        total_params = sum(p.size for p in m.parameters())
        # Rough estimate: 4 bytes per parameter for float32
        return total_params * 4 / (1024 ** 2)

    original_size = model_size_mb(model)

    for bits in bits_list:
        logger.info(f"Benchmarking {bits}-bit quantization...")

        # Quantize
        start = time.time()
        q_model = quantize_model(model, bits=bits)
        quant_time = time.time() - start

        # Estimate size (rough approximation)
        size_reduction = (1 - bits / 32) * 100

        results[f"{bits}-bit"] = {
            'quantization_time': quant_time,
            'size_reduction': size_reduction,
        }

        logger.info(f"  Quantization time: {quant_time:.2f}s")
        logger.info(f"  Estimated size reduction: {size_reduction:.1f}%")

    return results
