# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
# MIT License (https://opensource.org/licenses/MIT)
# Modified from 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker)
# MLX port for Apple Silicon

"""
XVector Speaker Encoder MLX Implementation

CAMPPlus (Context-Aware Masking Plus) speaker embedding extractor.
Uses dense TDNN blocks with context-aware masking for speaker verification.

Note: This is a simplified inference-only implementation. Training features
like checkpointing are not needed for inference.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import List


def statistics_pooling(
    x: mx.array, dim: int = -1, keepdim: bool = False, eps: float = 1e-2
) -> mx.array:
    """
    Compute mean and std statistics for pooling.

    Args:
        x: Input tensor
        dim: Dimension to compute statistics over
        keepdim: Whether to keep the dimension
        eps: Small value for numerical stability

    Returns:
        Concatenated mean and std
    """
    mean = mx.mean(x, axis=dim)
    # Compute std manually: sqrt(E[x^2] - E[x]^2)
    var = mx.mean(x * x, axis=dim) - mean * mean
    std = mx.sqrt(mx.maximum(var, eps * eps))

    stats = mx.concatenate([mean, std], axis=-1)

    if keepdim:
        stats = mx.expand_dims(stats, axis=dim)

    return stats


class BasicResBlockMLX(nn.Module):
    """Basic residual block for FCM."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()

        # Main path
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm(planes)

        # Shortcut
        self.use_shortcut = stride != 1 or in_planes != self.expansion * planes
        if self.use_shortcut:
            self.shortcut_conv = nn.Conv2d(
                in_planes,
                self.expansion * planes,
                kernel_size=1,
                stride=(stride, 1),
                bias=False,
            )
            self.shortcut_bn = nn.BatchNorm(self.expansion * planes)

    def __call__(self, x: mx.array) -> mx.array:
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.use_shortcut:
            shortcut = self.shortcut_bn(self.shortcut_conv(x))
        else:
            shortcut = x

        out = out + shortcut
        out = nn.relu(out)
        return out


class FCMMLX(nn.Module):
    """
    Frequency-Channel Mapping module.

    Applies 2D convolutions to process frequency-channel features.
    """

    def __init__(
        self, m_channels: int = 32, feat_dim: int = 80, num_blocks: List[int] = [2, 2]
    ):
        super().__init__()

        self.in_planes = m_channels

        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm(m_channels)

        # Build residual layers with explicit attributes
        # PyTorch uses layer1.0, layer1.1, layer2.0, layer2.1 naming
        self._make_layer_named("layer1", m_channels, num_blocks[0], stride=2)
        self._make_layer_named("layer2", m_channels, num_blocks[1], stride=2)

        self.conv2 = nn.Conv2d(
            m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm(m_channels)

        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer_named(self, prefix: str, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        for i, s in enumerate(strides):
            block = BasicResBlockMLX(self.in_planes, planes, s)
            setattr(self, f"{prefix}_{i}", block)
            self.in_planes = planes * BasicResBlockMLX.expansion
        # Store the number of blocks for iteration
        setattr(self, f"{prefix}_num_blocks", num_blocks)

    def __call__(self, x: mx.array) -> mx.array:
        # Add channel dim: [B, F, T] -> [B, 1, F, T]
        x = mx.expand_dims(x, axis=1)

        # MLX Conv2d expects [B, H, W, C] (NHWC), but we have [B, C, H, W] (NCHW)
        # Transpose to NHWC
        x = mx.transpose(x, axes=(0, 2, 3, 1))

        out = nn.relu(self.bn1(self.conv1(x)))

        for i in range(self.layer1_num_blocks):
            layer = getattr(self, f"layer1_{i}")
            out = layer(out)
        for i in range(self.layer2_num_blocks):
            layer = getattr(self, f"layer2_{i}")
            out = layer(out)

        out = nn.relu(self.bn2(self.conv2(out)))

        # out is [B, H, W, C] in NHWC format
        # PyTorch does: [B, C, H, W] -> reshape to [B, C*H, W]
        # We need to: [B, H, W, C] -> transpose to [B, C, H, W] -> reshape to [B, C*H, W]
        shape = out.shape  # [B, H, W, C]
        # Transpose to NCHW: [B, H, W, C] -> [B, C, H, W]
        out = mx.transpose(out, axes=(0, 3, 1, 2))  # [B, C, H, W]
        # Reshape to [B, C*H, W]
        out = mx.reshape(out, (shape[0], shape[3] * shape[1], shape[2]))

        return out


class TDNNLayerMLX(nn.Module):
    """Time Delay Neural Network layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
        use_batchnorm: bool = True,
        use_relu: bool = True,
    ):
        super().__init__()

        if padding < 0:
            assert kernel_size % 2 == 1, f"Expect odd kernel size, got {kernel_size}"
            padding = (kernel_size - 1) // 2 * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.use_batchnorm = use_batchnorm
        self.use_relu = use_relu

        if use_batchnorm:
            self.bn = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        if self.use_batchnorm:
            # BatchNorm expects [B, ..., C], Conv1d output is [B, T, C]
            x = self.bn(x)
        if self.use_relu:
            x = nn.relu(x)
        return x


class CAMLayerMLX(nn.Module):
    """Context-Aware Masking layer."""

    def __init__(
        self,
        bn_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        bias: bool,
        reduction: int = 2,
    ):
        super().__init__()

        self.linear_local = nn.Conv1d(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)

    def seg_pooling(
        self, x: mx.array, seg_len: int = 100, stype: str = "avg"
    ) -> mx.array:
        """Segment-level pooling.

        Mimics PyTorch's avg_pool1d with ceil_mode=True, which averages
        only over actual elements in partial windows (no zero padding).
        """
        # x: [B, T, C] after Conv1d
        B, T, C = x.shape

        # Number of segments
        num_segs = (T + seg_len - 1) // seg_len  # ceil division

        # Compute segment means handling partial last segment correctly
        segments = []
        for i in range(num_segs):
            start = i * seg_len
            end = min((i + 1) * seg_len, T)
            seg_slice = x[:, start:end, :]  # [B, actual_len, C]

            if stype == "avg":
                seg_mean = mx.mean(seg_slice, axis=1, keepdims=True)  # [B, 1, C]
            elif stype == "max":
                seg_mean = mx.max(seg_slice, axis=1, keepdims=True)  # [B, 1, C]
            else:
                raise ValueError(f"Wrong segment pooling type: {stype}")

            segments.append(seg_mean)

        # Stack segments: [B, num_segs, C]
        seg = mx.concatenate(segments, axis=1)  # [B, num_segs, C]

        # Expand back: [B, num_segs, C] -> [B, num_segs, 1, C] -> [B, num_segs, seg_len, C]
        seg = mx.expand_dims(seg, axis=2)
        seg = mx.repeat(seg, seg_len, axis=2)

        # Reshape: [B, num_segs * seg_len, C]
        seg = mx.reshape(seg, (B, -1, C))

        # Trim to original length
        seg = seg[:, :T, :]

        return seg

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, T, C]
        y = self.linear_local(x)  # [B, T, out_channels]

        # Global context: mean over time + segment pooling
        context = mx.mean(x, axis=1, keepdims=True) + self.seg_pooling(x)
        context = nn.relu(self.linear1(context))
        m = mx.sigmoid(self.linear2(context))

        return y * m


class CAMDenseTDNNLayerMLX(nn.Module):
    """CAM Dense TDNN Layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()

        assert kernel_size % 2 == 1, f"Expect odd kernel size, got {kernel_size}"
        padding = (kernel_size - 1) // 2 * dilation

        self.bn1 = nn.BatchNorm(in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm(bn_channels)

        self.cam_layer = CAMLayerMLX(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(nn.relu(self.bn1(x)))
        x = self.cam_layer(nn.relu(self.bn2(x)))
        return x


class CAMDenseTDNNBlockMLX(nn.Module):
    """CAM Dense TDNN Block - stacks multiple CAMDenseTDNNLayers."""

    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()

        self.num_layers = num_layers
        # Use explicit attributes for MLX parameter tracking
        # PyTorch uses tdnnd1, tdnnd2, ..., tdnnd{num_layers} naming
        for i in range(num_layers):
            layer = CAMDenseTDNNLayerMLX(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
            )
            # Match PyTorch naming: tdnnd1, tdnnd2, ..., tdnnd{num_layers}
            setattr(self, f"tdnnd{i + 1}", layer)

    def __call__(self, x: mx.array) -> mx.array:
        for i in range(self.num_layers):
            layer = getattr(self, f"tdnnd{i + 1}")
            x = mx.concatenate([x, layer(x)], axis=-1)
        return x


class TransitLayerMLX(nn.Module):
    """Transit layer for channel reduction."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.bn = nn.BatchNorm(in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.bn(x))
        x = self.linear(x)
        return x


class DenseLayerMLX(nn.Module):
    """Dense layer with batch normalization (uses Conv1d kernel_size=1 like PyTorch)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
        affine: bool = False,
    ):
        super().__init__()
        # Use Conv1d with kernel_size=1 to match PyTorch implementation
        self.linear = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm(out_channels, affine=affine)

    def __call__(self, x: mx.array) -> mx.array:
        # x can be [B, C] or [B, T, C] (MLX format: channels last)
        if len(x.shape) == 2:
            # [B, C] -> [B, 1, C] -> conv -> [B, 1, out_C] -> [B, out_C]
            x = x[:, None, :]  # [B, 1, C]
            x = self.linear(x)  # [B, 1, out_C]
            x = x.squeeze(1)  # [B, out_C]
        else:
            # [B, T, C] -> conv -> [B, T, out_C]
            x = self.linear(x)  # [B, T, out_C]
        x = self.bn(x)
        return x


class StatsPoolMLX(nn.Module):
    """Statistics pooling layer."""

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, T, C]
        return statistics_pooling(x, dim=1)


class CAMPPlusMLX(nn.Module):
    """
    CAMPPlus speaker embedding extractor.

    Uses CAM Dense TDNN blocks for extracting speaker embeddings.
    """

    def __init__(
        self,
        feat_dim: int = 80,
        embedding_size: int = 192,
        growth_rate: int = 32,
        bn_size: int = 4,
        init_channels: int = 128,
        output_level: str = "segment",
    ):
        super().__init__()

        self.output_level = output_level

        # FCM head
        self.head = FCMMLX(feat_dim=feat_dim)
        channels = self.head.out_channels

        # Initial TDNN layer
        self.tdnn = TDNNLayerMLX(
            channels, init_channels, kernel_size=5, stride=2, dilation=1, padding=-1
        )
        channels = init_channels

        # CAM Dense TDNN blocks and transit layers
        # Use explicit block attributes for MLX parameter tracking
        # PyTorch uses xvector.block1, xvector.block2, xvector.block3 naming
        for i, (num_layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2))
        ):
            block = CAMDenseTDNNBlockMLX(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            # Match PyTorch naming: block1, block2, block3
            setattr(self, f"block{i + 1}", block)

            channels = channels + num_layers * growth_rate

            transit = TransitLayerMLX(channels, channels // 2, bias=False)
            # Match PyTorch naming: transit1, transit2, transit3
            setattr(self, f"transit{i + 1}", transit)

            channels //= 2

        # Output nonlinearity
        self.out_bn = nn.BatchNorm(channels)

        if self.output_level == "segment":
            self.stats_pool = StatsPoolMLX()
            self.dense = DenseLayerMLX(channels * 2, embedding_size, affine=False)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input features [B, T, F] or [B, F, T]

        Returns:
            Speaker embedding [B, embedding_size] for segment level
        """
        # Transpose if needed: expect [B, T, F] input
        # FCM expects [B, F, T]
        if x.shape[-1] == 80:  # Assume F=80 at last dim means [B, T, F]
            x = mx.transpose(x, axes=(0, 2, 1))  # [B, F, T]

        # FCM head - outputs [B, H*C, T]
        x = self.head(x)

        # Transpose for Conv1d: [B, H*C, T] -> [B, T, H*C]
        x = mx.transpose(x, axes=(0, 2, 1))

        # Initial TDNN
        x = self.tdnn(x)

        # CAM Dense TDNN blocks with transit layers
        for i in range(3):
            block = getattr(self, f"block{i + 1}")
            transit = getattr(self, f"transit{i + 1}")
            x = block(x)
            x = transit(x)

        # Output nonlinearity
        x = nn.relu(self.out_bn(x))

        if self.output_level == "segment":
            x = self.stats_pool(x)  # [B, channels * 2]
            x = self.dense(x)  # [B, embedding_size]
        else:
            # Frame level output
            pass

        return x

    def inference(self, audio_features: mx.array) -> mx.array:
        """
        Inference method for speaker embedding extraction.

        Args:
            audio_features: Pre-computed fbank features [B, T, F]

        Returns:
            Speaker embeddings [B, embedding_size]
        """
        return self(audio_features)


def convert_xvector_weights(pytorch_state_dict: dict) -> dict:
    """
    Convert PyTorch CAMPPlus weights to MLX format.

    Args:
        pytorch_state_dict: PyTorch state dict for CAMPPlus

    Returns:
        Dictionary with MLX-compatible weights
    """
    mlx_weights = {}

    for key, value in pytorch_state_dict.items():
        numpy_val = value.numpy() if hasattr(value, "numpy") else value

        # Handle different layer types
        if "head." in key:
            # FCM module
            new_key = key.replace("head.", "head.")

            # Conv2d weights: PyTorch [out, in, H, W] -> MLX [out, H, W, in]
            if "conv" in key and "weight" in key and len(numpy_val.shape) == 4:
                numpy_val = numpy_val.transpose(0, 2, 3, 1)

        elif "xvector." in key:
            # Main xvector sequential module
            # Parse and map layer names
            new_key = key.replace("xvector.", "")

            # Map tdnn to self.tdnn
            if new_key.startswith("tdnn."):
                new_key = "tdnn." + new_key[5:]
            # Map blocks
            elif new_key.startswith("block"):
                # block1, block2, block3 -> self.blocks[0,1,2]
                block_idx = int(new_key[5]) - 1
                rest = new_key[7:]  # after "block1."
                new_key = f"blocks.{block_idx}.{rest}"
            # Map transits
            elif new_key.startswith("transit"):
                transit_idx = int(new_key[7]) - 1
                rest = new_key[9:]  # after "transit1."
                new_key = f"transits.{transit_idx}.{rest}"
            # Map out_nonlinear (batchnorm)
            elif new_key.startswith("out_nonlinear."):
                new_key = "out_bn." + new_key[14:]
            # Map stats and dense
            elif new_key.startswith("stats."):
                continue  # StatsPool has no parameters
            elif new_key.startswith("dense."):
                new_key = "dense." + new_key[6:]
        else:
            new_key = key

        # Handle Conv1d weights: PyTorch [out, in, K] -> MLX [out, K, in]
        if "linear" in key or "conv" in key:
            if "weight" in key and len(numpy_val.shape) == 3:
                numpy_val = numpy_val.transpose(0, 2, 1)

        mlx_weights[new_key] = mx.array(numpy_val)

    return mlx_weights
