# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
KV Cache implementation for MLX-based autoregressive generation.
"""

from typing import List, Optional, Tuple
import mlx.core as mx


class KVCacheMLX:
    """
    Key-Value cache for efficient autoregressive generation in MLX.

    This class manages the KV cache across transformer layers, supporting:
    - Memory-efficient storage with configurable dtype
    - Easy updates and concatenation
    - Compatible with T3MLX generation pipeline
    """

    def __init__(self, num_layers: int = 0, dtype: mx.Dtype = mx.float16):
        """
        Initialize KV cache.

        Args:
            num_layers: Number of transformer layers (can be set later)
            dtype: Data type for cache (float16 for memory efficiency, float32 for precision)
        """
        self.dtype = dtype
        self.num_layers = num_layers
        self.cache: List[Optional[Tuple[mx.array, mx.array]]] = [None] * num_layers

    def update(self, layer_idx: int, key: mx.array, value: mx.array):
        """
        Update cache for a specific layer.

        Args:
            layer_idx: Index of the transformer layer
            key: Key tensor to cache
            value: Value tensor to cache
        """
        # Ensure we have enough cache slots
        while len(self.cache) <= layer_idx:
            self.cache.append(None)

        # Convert to specified dtype for memory optimization
        key = key.astype(self.dtype)
        value = value.astype(self.dtype)

        if self.cache[layer_idx] is None:
            # First update for this layer
            self.cache[layer_idx] = (key, value)
        else:
            # Concatenate with existing cache
            cached_key, cached_value = self.cache[layer_idx]
            self.cache[layer_idx] = (
                mx.concatenate([cached_key, key], axis=1),  # Concatenate along sequence dimension
                mx.concatenate([cached_value, value], axis=1)
            )

    def get(self, layer_idx: int) -> Optional[Tuple[mx.array, mx.array]]:
        """
        Get cache for a specific layer.

        Args:
            layer_idx: Index of the transformer layer

        Returns:
            Tuple of (key_cache, value_cache) or None if not cached
        """
        if layer_idx >= len(self.cache):
            return None
        return self.cache[layer_idx]

    def get_all(self) -> List[Optional[Tuple[mx.array, mx.array]]]:
        """
        Get all cached key-value pairs.

        Returns:
            List of (key, value) tuples for each layer
        """
        return self.cache

    def clear(self):
        """Clear all cached values."""
        self.cache = [None] * len(self.cache)

    def get_sequence_length(self, layer_idx: int = 0) -> int:
        """
        Get the current sequence length from cache.

        Args:
            layer_idx: Layer to check (default: 0)

        Returns:
            Sequence length or 0 if no cache
        """
        cache_entry = self.get(layer_idx)
        if cache_entry is None:
            return 0
        key_cache, _ = cache_entry
        return key_cache.shape[1]  # Sequence dimension

    def __len__(self) -> int:
        """Return number of layers in cache."""
        return len(self.cache)

    def __repr__(self) -> str:
        cached_layers = sum(1 for c in self.cache if c is not None)
        seq_len = self.get_sequence_length(0) if cached_layers > 0 else 0
        return f"KVCacheMLX(layers={len(self.cache)}, cached={cached_layers}, seq_len={seq_len}, dtype={self.dtype})"
