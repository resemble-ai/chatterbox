"""
Advanced quantization for TTS models (INT8/INT4)

Implements:
- Smooth Quantization for activations
- GPTQ-style weight quantization
- KV cache quantization
- Per-channel quantization
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SmoothQuantLinear(nn.Module):
    """
    Smooth Quantization for Linear layers

    Paper: SmoothQuant: Accurate and Efficient Post-Training Quantization
    https://arxiv.org/abs/2211.10438
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_bits: int = 8,
        act_bits: int = 8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bits = weight_bits
        self.act_bits = act_bits

        # Quantized weights (INT8)
        self.register_buffer(
            'weight_quantized',
            torch.zeros((out_features, in_features), dtype=torch.int8)
        )
        self.register_buffer('weight_scale', torch.ones(out_features))

        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.bias = None

        # Smoothing scales for activations
        self.register_buffer('smooth_scale', torch.ones(in_features))
        self.register_buffer('act_scale', torch.tensor(1.0))

    @staticmethod
    def from_float(
        module: nn.Linear,
        weight_bits: int = 8,
        act_bits: int = 8,
        alpha: float = 0.5,
    ) -> 'SmoothQuantLinear':
        """
        Convert a float Linear layer to SmoothQuant

        Args:
            module: Original nn.Linear layer
            weight_bits: Bits for weight quantization
            act_bits: Bits for activation quantization
            alpha: Smoothing factor (0=no smooth, 1=full smooth)
        """
        quant_linear = SmoothQuantLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            weight_bits,
            act_bits,
        )

        with torch.no_grad():
            weight = module.weight.data  # (out, in)

            # Compute smoothing scales
            # s = (max(|X|)^alpha / max(|W|)^(1-alpha))
            weight_max = weight.abs().max(dim=0)[0]  # Per input channel

            # For activation, we use a heuristic (would be better with calibration data)
            act_max = torch.ones_like(weight_max)

            smooth_scale = (act_max ** alpha) / (weight_max ** (1 - alpha) + 1e-5)
            smooth_scale = torch.clamp(smooth_scale, 0.01, 100)

            # Apply smoothing to weights
            smoothed_weight = weight * smooth_scale.unsqueeze(0)

            # Quantize weights per output channel
            weight_scale = smoothed_weight.abs().max(dim=1)[0] / 127.0
            weight_scale = torch.clamp(weight_scale, 1e-5, 1e5)

            weight_quantized = torch.clamp(
                torch.round(smoothed_weight / weight_scale.unsqueeze(1)),
                -128, 127
            ).to(torch.int8)

            quant_linear.weight_quantized = weight_quantized
            quant_linear.weight_scale = weight_scale
            quant_linear.smooth_scale = smooth_scale

            if module.bias is not None:
                quant_linear.bias = module.bias.data.clone()

        return quant_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with INT8 computation
        """
        # Apply smoothing to input
        x_smooth = x / self.smooth_scale

        # Quantize activations
        act_scale = x_smooth.abs().max() / 127.0
        x_quant = torch.clamp(
            torch.round(x_smooth / (act_scale + 1e-5)),
            -128, 127
        ).to(torch.int8)

        # INT8 matmul (will be cast to int32 internally)
        out = torch.nn.functional.linear(
            x_quant.to(torch.float32),
            self.weight_quantized.to(torch.float32),
        )

        # Dequantize
        out = out * act_scale * self.weight_scale.unsqueeze(0)

        if self.bias is not None:
            out = out + self.bias

        return out


class GPTQLinear(nn.Module):
    """
    GPTQ-style 4-bit weight quantization

    Paper: GPTQ: Accurate Post-Training Quantization for GPT
    https://arxiv.org/abs/2210.17323
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 4,
        group_size: int = 128,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        # Pack weights into uint8 (2 weights per byte for 4-bit)
        assert in_features % group_size == 0, "in_features must be divisible by group_size"

        self.register_buffer(
            'qweight',
            torch.zeros((out_features, in_features // 2), dtype=torch.uint8)
        )

        # Scales per group
        num_groups = in_features // group_size
        self.register_buffer(
            'scales',
            torch.ones((out_features, num_groups))
        )

        # Zero points per group
        self.register_buffer(
            'zeros',
            torch.zeros((out_features, num_groups), dtype=torch.int8)
        )

        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.bias = None

    @staticmethod
    def quantize_weight(
        weight: torch.Tensor,
        bits: int = 4,
        group_size: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize weights to 4-bit with groupwise quantization
        """
        out_features, in_features = weight.shape
        assert in_features % group_size == 0

        num_groups = in_features // group_size

        # Reshape to groups
        weight_grouped = weight.reshape(out_features, num_groups, group_size)

        # Compute scales and zeros per group
        max_q = 2 ** bits - 1

        w_min = weight_grouped.min(dim=2, keepdim=True)[0]
        w_max = weight_grouped.max(dim=2, keepdim=True)[0]

        scales = (w_max - w_min) / max_q
        zeros = (-w_min / scales).round().clamp(0, max_q)

        # Quantize
        weight_q = ((weight_grouped / scales) + zeros).round().clamp(0, max_q)

        # Pack into uint8 (2 weights per byte)
        weight_q = weight_q.reshape(out_features, in_features)
        qweight = torch.zeros((out_features, in_features // 2), dtype=torch.uint8)

        for i in range(in_features // 2):
            qweight[:, i] = (
                weight_q[:, 2*i].to(torch.uint8) |
                (weight_q[:, 2*i+1].to(torch.uint8) << 4)
            )

        return qweight, scales.squeeze(2), zeros.squeeze(2)

    @staticmethod
    def from_float(
        module: nn.Linear,
        bits: int = 4,
        group_size: int = 128,
    ) -> 'GPTQLinear':
        """Convert float Linear to GPTQ 4-bit"""
        quant_linear = GPTQLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            bits,
            group_size,
        )

        with torch.no_grad():
            qweight, scales, zeros = GPTQLinear.quantize_weight(
                module.weight.data,
                bits,
                group_size,
            )

            quant_linear.qweight = qweight
            quant_linear.scales = scales
            quant_linear.zeros = zeros

            if module.bias is not None:
                quant_linear.bias = module.bias.data.clone()

        return quant_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantize and compute"""
        # Unpack weights
        weight = torch.zeros(
            (self.out_features, self.in_features),
            dtype=x.dtype,
            device=x.device,
        )

        for i in range(self.in_features // 2):
            weight[:, 2*i] = (self.qweight[:, i] & 0x0F).to(x.dtype)
            weight[:, 2*i+1] = ((self.qweight[:, i] >> 4) & 0x0F).to(x.dtype)

        # Dequantize with group scales
        num_groups = self.in_features // self.group_size
        weight = weight.reshape(self.out_features, num_groups, self.group_size)

        scales = self.scales.unsqueeze(2)
        zeros = self.zeros.unsqueeze(2)

        weight = (weight - zeros) * scales
        weight = weight.reshape(self.out_features, self.in_features)

        # Linear
        out = torch.nn.functional.linear(x, weight)

        if self.bias is not None:
            out = out + self.bias

        return out


class KVCacheINT8:
    """
    INT8 quantized KV cache for memory efficiency

    Reduces KV cache memory by 4x (FP32 -> INT8)
    """
    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        device: str = "cuda",
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device

        # Allocate INT8 cache
        self.k_cache = torch.zeros(
            (num_layers, max_batch_size, num_heads, max_seq_len, head_dim),
            dtype=torch.int8,
            device=device,
        )
        self.v_cache = torch.zeros(
            (num_layers, max_batch_size, num_heads, max_seq_len, head_dim),
            dtype=torch.int8,
            device=device,
        )

        # Scales for dequantization
        self.k_scales = torch.ones(
            (num_layers, max_batch_size, num_heads, max_seq_len),
            dtype=torch.float32,
            device=device,
        )
        self.v_scales = torch.ones(
            (num_layers, max_batch_size, num_heads, max_seq_len),
            dtype=torch.float32,
            device=device,
        )

        self.seq_lens = torch.zeros(max_batch_size, dtype=torch.long, device=device)

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,  # (batch, num_heads, seq_len, head_dim)
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new K/V and return full cached K/V"""
        batch_size, num_heads, new_seq_len, head_dim = k.shape

        for b in range(batch_size):
            start_pos = self.seq_lens[b].item()
            end_pos = start_pos + new_seq_len

            # Quantize K
            k_b = k[b]  # (num_heads, new_seq_len, head_dim)
            k_scale = k_b.abs().max(dim=-1, keepdim=True)[0] / 127.0
            k_scale = torch.clamp(k_scale, 1e-5, 1e5)
            k_quant = torch.clamp(
                torch.round(k_b / k_scale),
                -128, 127
            ).to(torch.int8)

            # Store
            self.k_cache[layer_idx, b, :, start_pos:end_pos] = k_quant
            self.k_scales[layer_idx, b, :, start_pos:end_pos] = k_scale.squeeze(-1)

            # Quantize V
            v_b = v[b]
            v_scale = v_b.abs().max(dim=-1, keepdim=True)[0] / 127.0
            v_scale = torch.clamp(v_scale, 1e-5, 1e5)
            v_quant = torch.clamp(
                torch.round(v_b / v_scale),
                -128, 127
            ).to(torch.int8)

            self.v_cache[layer_idx, b, :, start_pos:end_pos] = v_quant
            self.v_scales[layer_idx, b, :, start_pos:end_pos] = v_scale.squeeze(-1)

            self.seq_lens[b] = end_pos

        # Dequantize and return full cache
        max_len = self.seq_lens.max().item()

        k_full = self.k_cache[layer_idx, :batch_size, :, :max_len].to(k.dtype)
        k_scales = self.k_scales[layer_idx, :batch_size, :, :max_len].unsqueeze(-1)
        k_full = k_full * k_scales

        v_full = self.v_cache[layer_idx, :batch_size, :, :max_len].to(v.dtype)
        v_scales = self.v_scales[layer_idx, :batch_size, :, :max_len].unsqueeze(-1)
        v_full = v_full * v_scales

        return k_full, v_full


def quantize_model_int8(model: nn.Module, alpha: float = 0.5) -> nn.Module:
    """
    Quantize all Linear layers in model to INT8 using SmoothQuant

    Args:
        model: PyTorch model
        alpha: Smoothing factor

    Returns:
        Quantized model
    """
    logger.info("Quantizing model to INT8...")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Skip very small layers
            if module.in_features < 64 or module.out_features < 64:
                continue

            # Replace with quantized version
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            quant_layer = SmoothQuantLinear.from_float(module, alpha=alpha)
            setattr(parent, child_name, quant_layer)

            logger.debug(f"Quantized {name}")

    logger.info("INT8 quantization complete")
    return model


def quantize_model_int4(model: nn.Module, group_size: int = 128) -> nn.Module:
    """
    Quantize all Linear layers to INT4 using GPTQ

    Args:
        model: PyTorch model
        group_size: Group size for quantization

    Returns:
        Quantized model
    """
    logger.info("Quantizing model to INT4...")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if input features is divisible by group_size
            if module.in_features % group_size != 0:
                continue

            # Skip very small layers
            if module.in_features < 128 or module.out_features < 128:
                continue

            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            quant_layer = GPTQLinear.from_float(module, group_size=group_size)
            setattr(parent, child_name, quant_layer)

            logger.debug(f"Quantized {name} to INT4")

    logger.info("INT4 quantization complete")
    return model
