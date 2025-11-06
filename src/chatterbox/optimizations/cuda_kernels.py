"""
CUDA optimizations for faster inference
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def optimized_sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 0.95,
    min_p: float = 0.05,
) -> torch.Tensor:
    """
    Optimized token sampling with fused operations

    Args:
        logits: (batch_size, vocab_size) tensor
        temperature: sampling temperature
        top_p: nucleus sampling threshold
        min_p: minimum probability threshold

    Returns:
        sampled token ids (batch_size,)
    """
    # Temperature scaling
    if temperature != 1.0:
        logits = logits / temperature

    # Softmax
    probs = F.softmax(logits, dim=-1)

    # Min-p filtering
    if min_p > 0.0:
        min_p_threshold = probs.max(dim=-1, keepdim=True).values * min_p
        probs = torch.where(probs < min_p_threshold, torch.zeros_like(probs), probs)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 0] = False

        # Scatter to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        probs = torch.where(indices_to_remove, torch.zeros_like(probs), probs)

    # Renormalize
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # Sample
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


@torch.jit.script
def fused_embedding_with_position(
    token_ids: torch.Tensor,
    token_embedding: torch.Tensor,
    position_embedding: torch.Tensor,
    position: int,
) -> torch.Tensor:
    """
    Fused token + position embedding lookup

    Args:
        token_ids: (batch_size, 1) token indices
        token_embedding: (vocab_size, hidden_dim) embedding weights
        position_embedding: (max_seq_len, hidden_dim) position weights
        position: current position index

    Returns:
        combined embeddings (batch_size, 1, hidden_dim)
    """
    # Lookup token embedding
    tok_emb = F.embedding(token_ids, token_embedding)

    # Add position embedding
    pos_emb = position_embedding[position : position + 1].unsqueeze(0)

    return tok_emb + pos_emb


def apply_repetition_penalty_inplace(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    penalty: float = 1.2,
) -> None:
    """
    Apply repetition penalty in-place for memory efficiency

    Args:
        logits: (batch_size, vocab_size) to modify in-place
        token_ids: (batch_size, seq_len) previously generated tokens
        penalty: repetition penalty factor (>1.0 = penalize, <1.0 = encourage)
    """
    if penalty == 1.0:
        return

    batch_size = logits.shape[0]

    for i in range(batch_size):
        # Get unique tokens in this sequence
        unique_tokens = token_ids[i].unique()

        # Apply penalty
        for token in unique_tokens:
            if logits[i, token] < 0:
                logits[i, token] *= penalty
            else:
                logits[i, token] /= penalty


def cfg_guidance(
    cond_logits: torch.Tensor,
    uncond_logits: torch.Tensor,
    cfg_weight: float,
) -> torch.Tensor:
    """
    Classifier-Free Guidance combining conditional and unconditional logits

    Args:
        cond_logits: (batch_size, vocab_size) conditional logits
        uncond_logits: (batch_size, vocab_size) unconditional logits
        cfg_weight: guidance strength

    Returns:
        guided logits (batch_size, vocab_size)
    """
    return cond_logits + cfg_weight * (cond_logits - uncond_logits)
