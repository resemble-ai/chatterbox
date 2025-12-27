# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
MLX utility functions for mask operations.
Port of PyTorch mask utilities from s3gen/utils/mask.py
"""

import mlx.core as mx


def make_pad_mask(lengths: mx.array, max_len: int = 0) -> mx.array:
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (mx.array): Batch of lengths (B,).
        max_len (int): Maximum length. If 0, use max of lengths.

    Returns:
        mx.array: Mask tensor (B, max_len) where True indicates padding.

    Examples:
        >>> lengths = mx.array([5, 3, 2])
        >>> make_pad_mask(lengths)
        # masks = [[False, False, False, False, False],
        #          [False, False, False, True, True],
        #          [False, False, True, True, True]]
    """
    batch_size = lengths.shape[0]
    max_len = max_len if max_len > 0 else int(mx.max(lengths))
    seq_range = mx.arange(max_len)
    seq_range_expand = mx.broadcast_to(seq_range, (batch_size, max_len))
    seq_length_expand = mx.expand_dims(lengths, axis=-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def subsequent_mask(size: int) -> mx.array:
    """Create a causal (lower triangular) mask.

    Args:
        size: Size of the square mask.

    Returns:
        Lower triangular mask of shape (size, size) where True means attend.
    """
    pos_idx = mx.arange(size)
    # mask[i, j] = i >= j (lower triangular)
    mask = mx.expand_dims(pos_idx, axis=1) >= mx.expand_dims(pos_idx, axis=0)
    return mask


def subsequent_chunk_mask(
    size: int,
    chunk_size: int,
    num_left_chunks: int = -1,
) -> mx.array:
    """Create mask for subsequent steps (size, size) with chunk size.

    This is for streaming encoder.

    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks

    Returns:
        mx.array: mask of shape (size, size)

    Examples:
        >>> subsequent_chunk_mask(4, 2)
        # [[True, True, False, False],
        #  [True, True, False, False],
        #  [True, True, True, True],
        #  [True, True, True, True]]
    """
    pos_idx = mx.arange(size)
    block_value = ((pos_idx // chunk_size) + 1) * chunk_size
    ret = mx.expand_dims(pos_idx, axis=0) < mx.expand_dims(block_value, axis=1)
    return ret


def add_optional_chunk_mask(
    xs: mx.array,
    masks: mx.array,
    use_dynamic_chunk: bool,
    use_dynamic_left_chunk: bool,
    decoding_chunk_size: int,
    static_chunk_size: int,
    num_decoding_left_chunks: int,
    enable_full_context: bool = True,
) -> mx.array:
    """Apply optional mask for encoder.

    Args:
        xs (mx.array): padded input, (B, L, D), L for max length
        masks (mx.array): mask for xs, (B, 1, L)
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        use_dynamic_left_chunk (bool): whether to use dynamic left chunk
        decoding_chunk_size (int): decoding chunk size for dynamic chunk
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        static_chunk_size (int): chunk size for static chunk
        num_decoding_left_chunks: number of left chunks

    Returns:
        mx.array: chunk mask of the input xs.
    """
    # For inference, we typically use full context (no dynamic chunk)
    if use_dynamic_chunk:
        max_len = xs.shape[1]
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            # For inference we use full context
            chunk_size = max_len
            num_left_chunks = -1

        chunk_masks = subsequent_chunk_mask(xs.shape[1], chunk_size, num_left_chunks)
        chunk_masks = mx.expand_dims(chunk_masks, axis=0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(
            xs.shape[1], static_chunk_size, num_left_chunks
        )
        chunk_masks = mx.expand_dims(chunk_masks, axis=0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    else:
        chunk_masks = masks

    return chunk_masks


def mask_to_bias(mask: mx.array, dtype: mx.Dtype) -> mx.array:
    """Convert boolean mask to attention bias.

    Args:
        mask: Boolean mask where True means valid (attend)
        dtype: Output dtype

    Returns:
        Attention bias where invalid positions have large negative values
    """
    mask = mask.astype(dtype)
    # attention mask bias: 0 for valid, large negative for invalid
    mask = (1.0 - mask) * -1.0e10
    return mask
