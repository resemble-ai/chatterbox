# Copyright (c) 2025 MichaelYangAI
# MIT License

from typing import Union
import mlx.core as mx
import mlx.nn as nn


class LearnedPositionEmbeddingsMLX(nn.Module):
    """
    MLX implementation of learned position embeddings.
    Converted from PyTorch version in learned_pos_emb.py
    """

    def __init__(self, seq_len: int, model_dim: int, init: float = 0.02):
        """
        Initialize learned position embeddings.

        Args:
            seq_len: Maximum sequence length
            model_dim: Dimension of the model
            init: Standard deviation for initialization
        """
        super().__init__()
        self.seq_len = seq_len
        self.model_dim = model_dim

        # MLX Embedding layer
        self.emb = nn.Embedding(seq_len, model_dim)

        # Initialize weights with normal distribution
        # MLX initializes embeddings with uniform by default, so we override
        self.emb.weight = mx.random.normal(
            shape=(seq_len, model_dim), loc=0.0, scale=init
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Returns positional embeddings for index 0 up to the length of x.

        Args:
            x: Input tensor of shape (B, L, ...) or (L, ...)

        Returns:
            Positional embeddings of shape (L, model_dim)
        """
        sl = x.shape[1] if x.ndim > 1 else x.shape[0]
        indices = mx.arange(sl)
        return self.emb(indices)

    def get_fixed_embedding(self, idx: Union[int, mx.array]) -> mx.array:
        """
        Get positional embeddings for specific indices.

        Args:
            idx: Scalar int or integer array of shape (T,) or (B, T)

        Returns:
            Positional embeddings of shape (B, T, dim) or (1, 1, dim) for int input
        """
        # Convert int to array
        if isinstance(idx, int):
            idx = mx.array([[idx]])
        elif not isinstance(idx, mx.array):
            idx = mx.array(idx)

        # Ensure 2D
        if idx.ndim == 1:
            idx = mx.expand_dims(idx, 0)

        assert idx.ndim == 2, f"Expected 2D indices, got {idx.ndim}D"

        return self.emb(idx)  # (B, T, dim)
