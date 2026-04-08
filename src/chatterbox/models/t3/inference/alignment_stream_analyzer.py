# Copyright (c) 2025 Resemble AI
# Author: John Meade, Jeremy Hsu
# MIT License
import copy
import logging
import torch
from dataclasses import dataclass
from types import MethodType


logger = logging.getLogger(__name__)


LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]


@dataclass
class AlignmentAnalysisResult:
    # was this frame detected as being part of a noisy beginning chunk with potential hallucinations?
    false_start: bool
    # was this frame detected as being part of a long tail with potential hallucinations?
    long_tail: bool
    # was this frame detected as repeating existing text content?
    repetition: bool
    # was the alignment position of this frame too far from the previous frame?
    discontinuity: bool
    # has inference reached the end of the text tokens? eg, this remains false if inference stops early
    complete: bool
    # approximate position in the text token sequence. Can be used for generating online timestamps.
    position: int


class AlignmentStreamAnalyzer:
    def __init__(self, tfmr, queue, text_tokens_slice, alignment_layer_idx=9, eos_idx=0):
        """
        Some transformer TTS models implicitly solve text-speech alignment in one or more of their self-attention
        activation maps. This module exploits this to perform online integrity checks which streaming.
        A hook is injected into the specified attention layer, and heuristics are used to determine alignment
        position, repetition, etc.

        NOTE: currently requires no queues.
        """
        # self.queue = queue
        self.text_tokens_slice = (i, j) = text_tokens_slice
        self.eos_idx = eos_idx
        self.device = next(tfmr.parameters()).device
        self.alignment = torch.zeros(0, j-i, device=self.device)
        # self.alignment_bin = torch.zeros(0, j-i)
        self.curr_frame_pos = 0
        self.text_position = 0

        self.started = False
        self.started_at = None

        self.complete = False
        self.completed_at = None
        
        # Track generated tokens for repetition detection
        self.generated_tokens = []

        # Using `output_attentions=True` globally is incompatible with optimized attention
        # kernels (SDPA/FlashAttention) — it forces ALL layers to eager math. Instead, we
        # inject output_attentions=True only on the 3 layers we need (9, 12, 13) via per-layer
        # pre-hooks, keeping the other 27 layers on fast SDPA.
        self.last_aligned_attns = []

        for i, (layer_idx, head_idx) in enumerate(LLAMA_ALIGNED_HEADS):
            self.last_aligned_attns += [None]
            self._add_attention_spy(tfmr, i, layer_idx, head_idx)

    def _add_attention_spy(self, tfmr, buffer_idx, layer_idx, head_idx):
        """
        Adds a forward hook to a specific attention layer to collect attention weights,
        and a pre-hook to force output_attentions=True only for this layer (keeping
        the other layers on fast SDPA).
        """
        def attention_forward_hook(module, input, output):
            """
            See `LlamaAttention.forward`; the output is a 3-tuple: `attn_output, attn_weights, past_key_value`.
            When `output_attentions=True`, the layer falls back to eager math and returns weights.
            `attn_output` has shape [B, H, T0, T0] for the 0th entry, and [B, H, 1, T0+i] for the rest i-th.
            """
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                step_attention = output[1]  # (B, n_heads, T0, Ti) — keep on GPU
                self.last_aligned_attns[buffer_idx] = step_attention[0, head_idx]  # (T0, Ti)

        def force_output_attentions(module, args, kwargs):
            """Pre-hook: inject output_attentions=True into this layer's forward kwargs."""
            kwargs['output_attentions'] = True
            return args, kwargs

        target_layer = tfmr.layers[layer_idx].self_attn

        # Give this layer its own config copy with eager attention so it can
        # return attention weights. All other layers keep the shared config (SDPA).
        eager_config = copy.copy(tfmr.config)
        eager_config._attn_implementation = 'eager'
        target_layer.config = eager_config

        target_layer.register_forward_pre_hook(force_output_attentions, with_kwargs=True)
        target_layer.register_forward_hook(attention_forward_hook)

    def step(self, logits, next_token=None):
        """
        Emits an AlignmentAnalysisResult into the output queue, and potentially modifies the logits to force an EOS.
        """
        # extract approximate alignment matrix chunk (1 frame at a time after the first chunk)
        aligned_attn = torch.stack(self.last_aligned_attns).mean(dim=0) # (N, N)
        i, j = self.text_tokens_slice
        if self.curr_frame_pos == 0:
            # first chunk has conditioning info, text tokens, and BOS token
            A_chunk = aligned_attn[j:, i:j].clone() # (T, S) — stays on GPU
        else:
            # subsequent chunks have 1 frame due to KV-caching
            A_chunk = aligned_attn[:, i:j].clone() # (1, S) — stays on GPU

        # TODO: monotonic masking; could have issue b/c spaces are often skipped.
        A_chunk[:, self.curr_frame_pos + 1:] = 0


        self.alignment = torch.cat((self.alignment, A_chunk), dim=0)

        A = self.alignment
        T, S = A.shape

        # update position
        cur_text_posn = A_chunk[-1].argmax()
        discontinuity = not(-4 < cur_text_posn - self.text_position < 7) # NOTE: very lenient!
        if not discontinuity:
            self.text_position = cur_text_posn

        # Hallucinations at the start of speech show up as activations at the bottom of the attention maps!
        # To mitigate this, we just wait until there are no activations far off-diagonal in the last 2 tokens,
        # and there are some strong activations in the first few tokens.
        false_start = (not self.started) and (A[-2:, -2:].max() > 0.1 or A[:, :4].max() < 0.5)
        self.started = not false_start
        if self.started and self.started_at is None:
            self.started_at = T

        # Is generation likely complete?
        self.complete = self.complete or self.text_position >= S - 3
        if self.complete and self.completed_at is None:
            self.completed_at = T

        # NOTE: EOS rarely assigned activations, and second-last token is often punctuation, so use last 3 tokens.
        # NOTE: due to the false-start behaviour, we need to make sure we skip activations for the first few tokens.
        last_text_token_duration = A[15:, -3:].sum()

        # Activations for the final token that last too long are likely hallucinations.
        long_tail = self.complete and S >= 3 and (A[self.completed_at:, -3:].sum(dim=0).max() >= 5) # 200ms

        # If there are activations in previous tokens after generation has completed, assume this is a repetition error.
        # Guard S > 5: A[:, :-5] produces an empty tensor when the alignment matrix has fewer than 5 columns
        # (short inputs), and max() on an empty reduction dim raises IndexError.
        alignment_repetition = self.complete and S > 5 and (A[self.completed_at:, :-5].max(dim=1).values.sum() > 5)
        
        # Track generated tokens for repetition detection
        if next_token is not None:
            # Convert tensor to scalar if needed
            if isinstance(next_token, torch.Tensor):
                token_id = next_token.item() if next_token.numel() == 1 else next_token.view(-1)[0].item()
            else:
                token_id = next_token
            self.generated_tokens.append(token_id)
            
            # Keep only last 8 tokens to prevent memory issues
            if len(self.generated_tokens) > 8:
                self.generated_tokens = self.generated_tokens[-8:]
            
        # Check for excessive token repetition (3x same token in a row)
        token_repetition = (
            # self.complete and
            len(self.generated_tokens) >= 3 and
            len(set(self.generated_tokens[-3:])) == 1
        )
        
        if token_repetition:
            repeated_token = self.generated_tokens[-1]
            logger.warning(f"🚨 Detected 2x repetition of token {repeated_token}")
            
        # Suppress EoS to prevent early termination
        if cur_text_posn < S - 3 and S > 5:  # Only suppress if text is longer than 5 tokens
            logits[..., self.eos_idx] = -2**15

        # If a bad ending is detected, force emit EOS by modifying logits
        # NOTE: this means logits may be inconsistent with latents!
        if long_tail or alignment_repetition or token_repetition:
            logger.warning(f"forcing EOS token, {long_tail=}, {alignment_repetition=}, {token_repetition=}")
            # (±2**15 is safe for all dtypes >= 16bit)
            logits = -(2**15) * torch.ones_like(logits)
            logits[..., self.eos_idx] = 2**15

        self.curr_frame_pos += 1
        return logits
