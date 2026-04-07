"""KV cache with max_position truncation for bucketed CUDA graph inference."""
from transformers import StaticCache


class KVTruncatingStaticCache(StaticCache):
    """StaticCache that truncates the returned key/value states to ``_max_position``.

    During CUDA graph capture, ``T3HuggingfaceBackend.forward()`` sets
    ``cache._max_position = <bucket_size>`` before calling the model.  This
    ``update()`` override then slices the returned KV tensors so that each
    attention layer only attends to the *filled* portion of the cache (rounded
    up to the nearest bucket), rather than the full ``max_cache_len`` allocation.

    The truncation is a CUDA op and gets baked into the captured graph at
    capture time.  During replay the same fixed-size slice is executed, which
    is correct because each bucket has its own captured graph.
    """

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        key_states, value_states = super().update(
            key_states, value_states, layer_idx, cache_kwargs
        )
        max_pos = getattr(self, "_max_position", None)
        if max_pos is not None:
            key_states = key_states[:, :, :max_pos]
            value_states = value_states[:, :, :max_pos]
        return key_states, value_states
