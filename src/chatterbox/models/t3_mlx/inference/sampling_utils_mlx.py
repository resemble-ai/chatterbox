# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
Sampling utilities for MLX-based generation.
Implements top-p, min-p, repetition penalty, and other sampling strategies.
"""

from typing import List
import mlx.core as mx


def apply_repetition_penalty(logits: mx.array, generated_ids: List[mx.array], penalty: float) -> mx.array:
    """
    Apply repetition penalty to logits.

    Tokens that have been generated before are penalized:
    - If logit < 0: multiply by penalty (make more negative)
    - If logit > 0: divide by penalty (make smaller)

    Args:
        logits: Logits tensor of shape (B, vocab_size)
        generated_ids: List of previously generated token tensors
        penalty: Penalty factor (> 1.0 penalizes repetition)

    Returns:
        Penalized logits of same shape
    """
    if len(generated_ids) == 0 or penalty == 1.0:
        return logits

    # Flatten all generated IDs
    all_ids = mx.concatenate([mx.reshape(t, [-1]) for t in generated_ids], axis=0)

    # Get unique token IDs that have been generated
    # MLX doesn't have unique(), so we'll implement a simple version
    all_ids_list = all_ids.tolist() if hasattr(all_ids, 'tolist') else [int(x) for x in all_ids]
    unique_ids = list(set(all_ids_list))

    # Apply penalty to each previously generated token
    for idx in unique_ids:
        # Get current logit value
        current_logit = logits[:, idx]

        # Apply penalty based on sign
        penalized = mx.where(
            current_logit < 0,
            current_logit * penalty,  # Make negative more negative
            current_logit / penalty   # Make positive smaller
        )

        # Update logits (MLX arrays support item assignment)
        logits = mx.concatenate([
            logits[:, :idx],
            mx.expand_dims(penalized, 1),
            logits[:, idx+1:]
        ], axis=1)

    return logits


def apply_top_p(logits: mx.array, top_p: float) -> mx.array:
    """
    Apply nucleus (top-p) sampling.

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
    sorted_probs = mx.sort(probs, axis=-1)[:, ::-1]  # Reverse to get descending

    # Compute cumulative probabilities
    cumsum_probs = mx.cumsum(sorted_probs, axis=-1)

    # Find cutoff: first position where cumsum > top_p
    # Shift cumsum right by 1 to include the token that crosses threshold
    cumsum_shifted = mx.concatenate([mx.zeros((cumsum_probs.shape[0], 1)), cumsum_probs[:, :-1]], axis=1)

    # Create mask: keep tokens where shifted cumsum <= top_p
    mask = cumsum_shifted <= top_p

    # Get sorted indices
    sorted_indices = mx.argsort(probs, axis=-1)[:, ::-1]

    # Apply mask to sorted probabilities
    sorted_probs_filtered = mx.where(mask, sorted_probs, 0.0)

    # Create a threshold: minimum probability to keep
    # Find the minimum non-zero value in each row
    B, V = logits.shape
    filtered_logits = logits.copy() if hasattr(logits, 'copy') else logits

    for b in range(B):
        # Get the probabilities we're keeping
        kept_probs = sorted_probs_filtered[b]
        min_kept_prob = mx.min(mx.where(kept_probs > 0, kept_probs, float('inf')))

        # Mask out probabilities below threshold
        row_mask = probs[b] >= min_kept_prob
        filtered_logits[b] = mx.where(row_mask, logits[b], -float('inf'))

    return filtered_logits


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
    filtered_logits = mx.where(mask, logits, -float('inf'))

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
    filtered_logits = mx.where(mask, logits, -float('inf'))

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
