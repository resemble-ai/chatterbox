# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Kai Hu)
# MLX port for Apple Silicon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
F0 Predictor MLX Implementation

Predicts fundamental frequency (F0) from mel-spectrograms using a CNN network.
"""

import mlx.core as mx
import mlx.nn as nn


class ConvRNNF0PredictorMLX(nn.Module):
    """
    Convolutional F0 predictor network.

    Uses a series of 1D convolutions with ELU activation to predict F0 values
    from mel-spectrogram features.

    Note: The original uses weight_norm which we implement as regular Conv1d.
    Weight normalization can be baked into the weights during conversion.
    """

    def __init__(
        self, num_class: int = 1, in_channels: int = 80, cond_channels: int = 512
    ):
        super().__init__()

        self.num_class = num_class
        self.in_channels = in_channels
        self.cond_channels = cond_channels

        # Conditioning network: 5 conv layers with ELU
        # Original uses weight_norm - we use regular Conv1d
        # Weight norm parameters (weight_g, weight_v) will be merged during conversion
        self.conv1 = nn.Conv1d(in_channels, cond_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)

        # Final classifier
        self.classifier = nn.Linear(cond_channels, num_class)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass of F0 predictor.

        Args:
            x: Input mel-spectrogram features [B, C, T] (channels first, PyTorch format)
               Will be transposed internally to MLX format [B, T, C]

        Returns:
            Predicted F0 values [B, T]
        """
        # MLX Conv1d expects [B, T, C] but PyTorch uses [B, C, T]
        # Transpose from PyTorch format to MLX format
        x = mx.transpose(x, axes=(0, 2, 1))  # [B, C, T] -> [B, T, C]

        # Apply conditioning network (conv layers with ELU)
        x = nn.elu(self.conv1(x))
        x = nn.elu(self.conv2(x))
        x = nn.elu(self.conv3(x))
        x = nn.elu(self.conv4(x))
        x = nn.elu(self.conv5(x))

        # x is now [B, T, C] which is what Linear expects
        # Classify and take absolute value
        x = mx.abs(self.classifier(x).squeeze(-1))

        return x


def convert_f0_predictor_weights(pytorch_state_dict: dict) -> dict:
    """
    Convert PyTorch F0 predictor weights to MLX format.

    Handles weight_norm by computing the actual weight from weight_g and weight_v.

    Args:
        pytorch_state_dict: PyTorch state dict for ConvRNNF0Predictor

    Returns:
        Dictionary with MLX-compatible weights
    """
    mlx_weights = {}

    # Map from PyTorch condnet Sequential indices to MLX conv layers
    conv_mapping = {
        "0": "conv1",  # First weight_norm conv
        "2": "conv2",  # Second weight_norm conv
        "4": "conv3",  # Third weight_norm conv
        "6": "conv4",  # Fourth weight_norm conv
        "8": "conv5",  # Fifth weight_norm conv
    }

    for key, value in pytorch_state_dict.items():
        if "condnet" in key:
            # Parse the sequential index
            parts = key.split(".")
            seq_idx = parts[1]

            if seq_idx in conv_mapping:
                mlx_key_prefix = conv_mapping[seq_idx]

                if "weight_g" in key:
                    # Store for later combination with weight_v
                    g_key = f"_g_{mlx_key_prefix}"
                    mlx_weights[g_key] = value.numpy()
                elif "weight_v" in key:
                    # Store for later combination with weight_g
                    v_key = f"_v_{mlx_key_prefix}"
                    mlx_weights[v_key] = value.numpy()
                elif "bias" in key:
                    # Conv1d bias: [C] stays as [C]
                    mlx_weights[f"{mlx_key_prefix}.bias"] = mx.array(value.numpy())
        elif "classifier" in key:
            if "weight" in key:
                # Linear weight: [out, in] -> [out, in] (MLX uses same format)
                mlx_weights["classifier.weight"] = mx.array(value.numpy())
            elif "bias" in key:
                mlx_weights["classifier.bias"] = mx.array(value.numpy())

    # Combine weight_g and weight_v to get actual weights
    # weight = weight_g * (weight_v / ||weight_v||)
    for conv_name in ["conv1", "conv2", "conv3", "conv4", "conv5"]:
        g_key = f"_g_{conv_name}"
        v_key = f"_v_{conv_name}"

        if g_key in mlx_weights and v_key in mlx_weights:
            weight_g = mlx_weights.pop(g_key)  # [out_channels, 1, 1]
            weight_v = mlx_weights.pop(
                v_key
            )  # [out_channels, in_channels, kernel_size]

            # Compute normalized weight
            import numpy as np

            norm = np.linalg.norm(
                weight_v.reshape(weight_v.shape[0], -1), axis=1, keepdims=True
            )
            norm = norm.reshape(weight_v.shape[0], 1, 1)
            weight = weight_g * (weight_v / (norm + 1e-12))

            # MLX Conv1d weight format: [out_channels, kernel_size, in_channels]
            # PyTorch format: [out_channels, in_channels, kernel_size]
            weight = np.transpose(weight, (0, 2, 1))
            mlx_weights[f"{conv_name}.weight"] = mx.array(weight)

    return mlx_weights
