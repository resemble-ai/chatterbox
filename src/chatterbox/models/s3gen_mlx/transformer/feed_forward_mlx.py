# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
MLX implementation of Positionwise Feed-Forward layer.
Port of PyTorch implementation from s3gen/transformer/positionwise_feed_forward.py
"""

import mlx.core as mx
import mlx.nn as nn


class PositionwiseFeedForwardMLX(nn.Module):
    """Positionwise feed forward layer for MLX.

    FeedForward is applied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimension.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate (not used in inference).
        activation (str): Activation function name.
    """

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float = 0.0,
        activation: str = "relu",
    ):
        """Construct a PositionwiseFeedForwardMLX object."""
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.w_2 = nn.Linear(hidden_units, idim)

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish" or activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

    def __call__(self, xs: mx.array) -> mx.array:
        """Forward function.

        Args:
            xs: Input tensor (B, L, D)

        Returns:
            Output tensor (B, L, D)
        """
        return self.w_2(self.activation(self.w_1(xs)))
