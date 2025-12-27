# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
Sampling utilities for MLX-based generation.
Implements top-p, min-p, repetition penalty, and other sampling strategies.
"""

from typing import List
import mlx.core as mx


def apply_repetition_penalty(
    logits: mx.array, generated_ids: List[mx.array], penalty: float
) -> mx.array:
    """
    Apply repetition penalty to logits (VECTORIZED).

    Tokens that have been generated before are penalized:
    - If logit < 0: multiply by penalty (make more negative)
    - If logit > 0: divide by penalty (make smaller)

    Args:
        logits: Logits tensor of shape (B, vocab_size)
        generated_ids: List of previously generated token IDs (can be ints or arrays)
        penalty: Penalty factor (> 1.0 penalizes repetition)

    Returns:
        Penalized logits of same shape
    """
    if len(generated_ids) == 0 or penalty == 1.0:
        return logits

    # Convert all IDs to integers (handles both int and mx.array)
    int_ids = []
    for t in generated_ids:
        if isinstance(t, int):
            int_ids.append(t)
        else:
            # It's an mx.array, convert to int
            int_ids.append(int(t.item() if t.size == 1 else t.reshape(-1)[0]))

    # Get unique token IDs using a set
    unique_ids = list(set(int_ids))

    if len(unique_ids) == 0:
        return logits

    # Create index array for scatter operation
    indices = mx.array(unique_ids, dtype=mx.int32)

    # Extract logits for penalized tokens: shape (B, num_unique)
    B, V = logits.shape
    penalized_logits = logits[:, indices]

    # Apply penalty based on sign (vectorized)
    penalized_values = mx.where(
        penalized_logits < 0,
        penalized_logits * penalty,  # Make negative more negative
        penalized_logits / penalty,  # Make positive smaller
    )

    # Create output by copying and updating in place using scatter
    # MLX supports at-indexing for vectorized updates
    result = logits.at[:, indices].add(penalized_values - penalized_logits)

    return result


def apply_top_p(logits: mx.array, top_p: float) -> mx.array:
    """
    Apply nucleus (top-p) sampling (VECTORIZED).

    Keeps only the most probable tokens whose cumulative probability >= top_p.

    Args:
        logits: Logits tensor of shape (B, vocab_size)
        top_p: Cumulative probability threshold (0.0 to 1.0)

    Returns:
        Filtered logits with low-probability tokens set to -inf
    """
    if top_p >= 1.0:
        return logits

    # Convert to probabilities
    probs = mx.softmax(logits, axis=-1)

    # Sort probabilities in descending order
    sorted_indices = mx.argsort(-probs, axis=-1)  # Descending by negating
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    # Compute cumulative probabilities
    cumsum_probs = mx.cumsum(sorted_probs, axis=-1)

    # Shift cumsum right by 1 to include the token that crosses threshold
    cumsum_shifted = mx.concatenate(
        [mx.zeros((cumsum_probs.shape[0], 1)), cumsum_probs[:, :-1]], axis=1
    )

    # Create mask: keep tokens where shifted cumsum <= top_p
    sorted_mask = cumsum_shifted <= top_p

    # Apply mask in sorted space
    sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
    sorted_logits_filtered = mx.where(sorted_mask, sorted_logits, -float("inf"))

    # Unsort back to original order
    # Create inverse permutation
    B, V = logits.shape
    mx.broadcast_to(mx.arange(B)[:, None], (B, V))

    # Scatter back to original positions
    result = mx.zeros_like(logits) - float("inf")  # Start with -inf

    # Use argsort of sorted_indices to get inverse permutation
    inverse_indices = mx.argsort(sorted_indices, axis=-1)
    result = mx.take_along_axis(sorted_logits_filtered, inverse_indices, axis=-1)

    return result


def apply_min_p(logits: mx.array, min_p: float) -> mx.array:
    """
    Apply min-p sampling.

    Filters out tokens whose probability is less than min_p * max_probability.

    Args:
        logits: Logits tensor of shape (B, vocab_size)
        min_p: Minimum probability threshold relative to max (0.0 to 1.0)

    Returns:
        Filtered logits with low-probability tokens set to -inf
    """
    if min_p <= 0.0:
        return logits

    # Convert to probabilities
    probs = mx.softmax(logits, axis=-1)

    # Get maximum probability for each batch
    max_prob = mx.max(probs, axis=-1, keepdims=True)

    # Compute threshold
    threshold = max_prob * min_p

    # Create mask for tokens above threshold
    mask = probs >= threshold

    # Apply mask
    filtered_logits = mx.where(mask, logits, -float("inf"))

    return filtered_logits


def apply_top_k(logits: mx.array, top_k: int) -> mx.array:
    """
    Apply top-k sampling.

    Keeps only the top k most probable tokens.

    Args:
        logits: Logits tensor of shape (B, vocab_size)
        top_k: Number of top tokens to keep

    Returns:
        Filtered logits with non-top-k tokens set to -inf
    """
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits

    # Get top-k values and indices
    top_k_logits = mx.topk(logits, top_k, axis=-1)

    # Get the k-th largest value (threshold)
    threshold = top_k_logits[:, -1:]

    # Mask out values below threshold
    mask = logits >= threshold
    filtered_logits = mx.where(mask, logits, -float("inf"))

    return filtered_logits


def sample_categorical(logits: mx.array, temperature: float = 1.0) -> mx.array:
    """
    Sample from categorical distribution with temperature.

    Args:
        logits: Logits tensor of shape (B, vocab_size)
        temperature: Sampling temperature (higher = more random)

    Returns:
        Sampled token indices of shape (B,)
    """
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Convert to probabilities
    probs = mx.softmax(logits, axis=-1)

    # Sample using categorical distribution
    samples = mx.random.categorical(mx.log(probs + 1e-10))

    return samples
