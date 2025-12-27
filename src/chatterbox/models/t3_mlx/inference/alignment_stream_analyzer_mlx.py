# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
MLX implementation of AlignmentStreamAnalyzer for quality control during generation.

Monitors attention patterns to detect and prevent:
- Long-tail hallucinations (generating after text is complete)
- Alignment-based repetition (re-speaking earlier text)
- Token repetition (same token repeated multiple times)
- Premature EOS termination
"""

import logging
import mlx.core as mx
from dataclasses import dataclass
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Specific Llama attention heads that show text-speech alignment
# Format: (layer_index, head_index)
LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]


@dataclass
class AlignmentAnalysisResult:
    """Results from alignment analysis at each generation step."""

    false_start: bool  # Noisy beginning with potential hallucinations
    long_tail: bool  # Generation continuing past text completion
    repetition: bool  # Repeating existing text content
    discontinuity: bool  # Alignment position jumped too far
    complete: bool  # Reached end of text tokens
    position: int  # Current position in text token sequence


class AlignmentStreamAnalyzerMLX:
    """
    MLX implementation of alignment-based quality control for TTS generation.

    This analyzer monitors self-attention patterns during autoregressive generation
    to detect when the model has completed speaking the input text and prevent
    quality issues like repetition and hallucination.

    Key differences from PyTorch version:
    - Uses MLX arrays instead of PyTorch tensors
    - Attention weights must be explicitly passed to step() method
    - No forward hooks (MLX doesn't support them like PyTorch)
    """

    def __init__(
        self,
        text_tokens_slice: Tuple[int, int],
        eos_idx: int = 0,
    ):
        """
        Initialize alignment stream analyzer.

        Args:
            text_tokens_slice: (start, end) indices of text tokens in the sequence
                              after conditioning. E.g., if conditioning is 100 tokens
                              and text is 50 tokens, slice would be (100, 150)
            eos_idx: Token ID for end-of-speech
        """
        self.text_tokens_slice = (i, j) = text_tokens_slice
        self.eos_idx = eos_idx

        # Alignment matrix: (T_speech, S_text) - tracks which text token each speech token aligns to
        self.alignment = mx.zeros((0, j - i))

        # Current frame position in speech sequence
        self.curr_frame_pos = 0

        # Current position in text sequence (argmax of alignment)
        self.text_position = 0

        # Has generation properly started (not in false-start phase)?
        self.started = False
        self.started_at: Optional[int] = None

        # Has generation completed all text tokens?
        self.complete = False
        self.completed_at: Optional[int] = None

        # Track recent generated tokens for repetition detection
        self.generated_tokens: List[int] = []

        # Buffer for attention weights from aligned heads
        # Will be populated by external code calling update_attention()
        self.last_aligned_attns: List[Optional[mx.array]] = [None] * len(
            LLAMA_ALIGNED_HEADS
        )

    def update_attention(
        self, attentions: List[mx.array], buffer_idx: int, layer_idx: int, head_idx: int
    ):
        """
        Update attention buffer with weights from a specific layer/head.

        Args:
            attentions: List of attention tensors from all layers, each (B, H, T, T)
            buffer_idx: Index in last_aligned_attns to update (0, 1, or 2)
            layer_idx: Transformer layer index
            head_idx: Attention head index within layer
        """
        if layer_idx < len(attentions) and attentions[layer_idx] is not None:
            # Extract specific head: attentions[layer] is (B, H, T, T)
            # We want (T, T) for batch 0, specific head
            attn_weights = attentions[layer_idx][0, head_idx, :, :]  # (T, T)
            self.last_aligned_attns[buffer_idx] = attn_weights

    def step(self, logits: mx.array, next_token: Optional[int] = None) -> mx.array:
        """
        Analyze alignment and modify logits to prevent quality issues.

        This is called at each generation step after computing logits but before sampling.
        It analyzes the attention patterns to detect issues and modifies logits to:
        - Force EOS if long tail / repetition / discontinuity detected
        - Suppress EOS if generation hasn't reached end of text yet

        Args:
            logits: Current logits tensor (1, vocab_size)
            next_token: Last generated token ID (for repetition tracking)

        Returns:
            Modified logits with EOS forced/suppressed as needed
        """
        # Check if we have attention data
        valid_attns = [a for a in self.last_aligned_attns if a is not None]
        if len(valid_attns) == 0:
            # No attention data available yet, skip analysis
            self.curr_frame_pos += 1
            return logits

        # Average attention across aligned heads
        aligned_attn = mx.stack(valid_attns).mean(axis=0)  # (T, T)

        i, j = self.text_tokens_slice

        # Extract alignment chunk for this step
        if self.curr_frame_pos == 0:
            # First chunk: includes conditioning, text, and BOS token
            # Shape: (num_speech_tokens, num_text_tokens)
            A_chunk = aligned_attn[j:, i:j]
        else:
            # Subsequent chunks: only 1 frame due to KV caching
            # Shape: (1, num_text_tokens)
            A_chunk = aligned_attn[-1:, i:j]

        # NOTE: Monotonic masking is intentionally disabled here.
        # The PyTorch version has a TODO noting this can cause issues because spaces are often skipped.
        # Masking was causing text_position to never advance, leading to infinite EOS suppression.
        # The alignment heuristics below handle position tracking without strict monotonic constraints.

        # Append to alignment matrix
        self.alignment = mx.concatenate([self.alignment, A_chunk], axis=0)

        A = self.alignment
        T, S = A.shape  # T = speech tokens, S = text tokens

        # Update text position based on alignment
        cur_text_posn = int(mx.argmax(A_chunk[-1]).item())

        # Check for discontinuity (position jumped too far)
        discontinuity = not (-4 < cur_text_posn - self.text_position < 7)
        if not discontinuity:
            self.text_position = cur_text_posn

        # Detect false start (hallucinations at beginning)
        # Wait until there are no activations far off-diagonal in last 2 tokens
        # and there are strong activations in first few tokens
        if T >= 2:
            false_start = (not self.started) and (
                float(mx.max(A[-2:, -2:]).item()) > 0.1
                or float(mx.max(A[:, :4]).item()) < 0.5
            )
        else:
            false_start = not self.started

        self.started = not false_start
        if self.started and self.started_at is None:
            self.started_at = T

        # Detect completion (reached end of text)
        self.complete = self.complete or self.text_position >= S - 3
        if self.complete and self.completed_at is None:
            self.completed_at = T

        # Detect long tail (generating after text is complete)
        long_tail = False
        if self.complete and T > self.completed_at:
            # Check if attention on final tokens continues for too long (>5 frames = 200ms)
            tail_attention = A[self.completed_at :, -3:]
            long_tail = float(mx.max(mx.sum(tail_attention, axis=0)).item()) >= 5

        # Detect alignment repetition (attention going back to earlier text)
        alignment_repetition = False
        if self.complete and T > self.completed_at:
            # Check if there are activations on earlier text tokens after completion
            past_attention = A[self.completed_at :, :-5]
            if past_attention.shape[1] > 0:
                alignment_repetition = (
                    float(mx.sum(mx.max(past_attention, axis=1)).item()) > 5
                )

        # Track token repetition
        if next_token is not None:
            self.generated_tokens.append(next_token)
            # Keep only last 8 tokens
            if len(self.generated_tokens) > 8:
                self.generated_tokens = self.generated_tokens[-8:]

        # Detect excessive token repetition
        # Use hybrid approach: stricter after completion, lenient during generation
        # Check if a single token appears too many times in recent history
        token_repetition = False
        repeated_token = None

        if self.complete:
            # After completion: 2x repetition is suspicious (matches PyTorch)
            # Check last 4 tokens for any token appearing >= 2 times
            if len(self.generated_tokens) >= 4:
                recent_tokens = self.generated_tokens[-4:]
                token_counts = {}
                for t in recent_tokens:
                    token_counts[t] = token_counts.get(t, 0) + 1
                max_count = max(token_counts.values())
                if max_count >= 2:
                    token_repetition = True
                    # Find the most repeated token
                    for t, count in token_counts.items():
                        if count == max_count:
                            repeated_token = t
                            break
        else:
            # During generation: look for a token appearing 4+ times in last 8 tokens
            # This catches patterns like "ya, ya, ya, ya" even with punctuation
            if len(self.generated_tokens) >= 6:
                recent_tokens = self.generated_tokens[-8:]
                token_counts = {}
                for t in recent_tokens:
                    token_counts[t] = token_counts.get(t, 0) + 1
                max_count = max(token_counts.values())
                if max_count >= 4:
                    token_repetition = True
                    # Find the most repeated token
                    for t, count in token_counts.items():
                        if count == max_count:
                            repeated_token = t
                            break

        if token_repetition and repeated_token is not None:
            logger.debug(f"🚨 Detected repetition of token {repeated_token}")

        # Suppress EOS before text completion (prevent premature termination)
        # Only suppress if text is reasonably long (> 5 tokens)
        if cur_text_posn < S - 3 and S > 5:
            # Create mask for EOS token position
            vocab_size = logits.shape[-1]
            eos_mask = mx.arange(vocab_size) == self.eos_idx
            # Set EOS logit to large negative value
            logits = mx.where(eos_mask[None, :], -32768.0, logits)

        # Force EOS if bad conditions detected
        if long_tail or alignment_repetition or token_repetition:
            logger.debug(
                f"Forcing EOS token: {long_tail=}, {alignment_repetition=}, {token_repetition=}"
            )
            # Set all logits to large negative
            vocab_size = logits.shape[-1]
            logits = mx.full((1, vocab_size), -32768.0)
            # Set EOS logit to large positive
            eos_mask = mx.arange(vocab_size) == self.eos_idx
            logits = mx.where(eos_mask[None, :], 32768.0, logits)

        self.curr_frame_pos += 1
        return logits

    def reset(self):
        """Reset analyzer state between generations."""
        i, j = self.text_tokens_slice
        self.alignment = mx.zeros((0, j - i))
        self.curr_frame_pos = 0
        self.text_position = 0
        self.started = False
        self.started_at = None
        self.complete = False
        self.completed_at = None
        self.generated_tokens = []
        self.last_aligned_attns = [None] * len(LLAMA_ALIGNED_HEADS)

    def get_analysis(self) -> AlignmentAnalysisResult:
        """
        Get current analysis results.

        Returns:
            AlignmentAnalysisResult with current state
        """
        # Check most recent step for issues
        discontinuity = False
        if self.alignment.shape[0] >= 2:
            last_pos = int(mx.argmax(self.alignment[-1]).item())
            prev_pos = int(mx.argmax(self.alignment[-2]).item())
            discontinuity = not (-4 < last_pos - prev_pos < 7)

        return AlignmentAnalysisResult(
            false_start=not self.started,
            long_tail=self.complete
            and self.curr_frame_pos - (self.completed_at or 0) > 5,
            repetition=len(self.generated_tokens) >= 2
            and len(set(self.generated_tokens[-2:])) == 1,
            discontinuity=discontinuity,
            complete=self.complete,
            position=self.text_position,
        )
