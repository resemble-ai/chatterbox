# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
T3 MLX Backend for generation.
MLX equivalent of T3HuggingfaceBackend from PyTorch version.
"""

from typing import Optional, Dict
import mlx.core as mx
import mlx.nn as nn


class T3MLXBackend(nn.Module):
    """
    MLX backend for T3 generation.
    Wraps the Llama model and adds custom speech embedding/projection layers.

    This is the MLX equivalent of T3HuggingfaceBackend, providing the interface
    needed for autoregressive generation with KV caching.
    """

    def __init__(
        self,
        llama_model: nn.Module,
        speech_emb: nn.Module,
        speech_head: nn.Module,
        config: object,
        alignment_stream_analyzer: Optional[object] = None,
    ):
        """
        Initialize T3 MLX backend.

        Args:
            llama_model: MLX Llama model
            speech_emb: Speech token embedding layer
            speech_head: Speech logit projection head
            config: Model configuration
            alignment_stream_analyzer: Optional alignment analyzer for multilingual models
        """
        super().__init__()
        self.model = llama_model
        self.speech_emb = speech_emb
        self.speech_head = speech_head
        self.config = config
        self.alignment_stream_analyzer = alignment_stream_analyzer
        self._added_cond = False

    def reset_state(self):
        """Reset generation state for new sequence."""
        self._added_cond = False
        if self.alignment_stream_analyzer is not None:
            self.alignment_stream_analyzer.reset()

        # Clear MLX cache to release GPU memory from previous generation
        mx.clear_cache()

    def __call__(
        self,
        inputs_embeds: mx.array,
        cache: Optional[list] = None,
        decoder_cond: Optional[mx.array] = None,
        use_cache: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = True,
    ) -> Dict:
        """
        Forward pass for generation.

        Args:
            inputs_embeds: Input embeddings of shape (B, L, D)
            cache: Optional KV cache from previous steps
            decoder_cond: Optional conditioning to prepend (used in first step)
            use_cache: Whether to return updated cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states

        Returns:
            Dictionary with 'logits', 'cache', 'hidden_states', etc.
        """
        # Prepend decoder conditioning on first step
        if decoder_cond is not None and not self._added_cond:
            # Expand conditioning if batch sizes don't match (for CFG)
            if decoder_cond.shape[0] != inputs_embeds.shape[0]:
                decoder_cond = mx.broadcast_to(
                    decoder_cond,
                    (
                        inputs_embeds.shape[0],
                        decoder_cond.shape[1],
                        decoder_cond.shape[2],
                    ),
                )
            inputs_embeds = mx.concatenate([decoder_cond, inputs_embeds], axis=1)
            self._added_cond = True

        # Forward through Llama model
        tfmr_out = self.model(
            inputs_embeds=inputs_embeds,
            cache=cache if use_cache else None,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        # Get final hidden states for logits projection
        # When output_hidden_states=False, hidden_states only contains final output
        hidden_states = tfmr_out["hidden_states"][-1]  # (B, seq, dim)

        # Project to speech logits
        logits = self.speech_head(hidden_states)

        result = {
            "logits": logits,
            "cache": tfmr_out.get("cache"),
            # Only include hidden_states if requested (saves memory during generation)
            "hidden_states": (
                tfmr_out.get("hidden_states") if output_hidden_states else None
            ),
            "last_hidden_state": hidden_states,
        }

        # Include attention weights if requested
        if output_attentions and "attentions" in tfmr_out:
            result["attentions"] = tfmr_out["attentions"]

        return result

    def prepare_inputs_for_generation(
        self,
        input_ids: mx.array,
        decoder_cond: mx.array,
        cache: Optional[list] = None,
    ) -> Dict:
        """
        Prepare inputs for generation step.

        Args:
            input_ids: Token IDs to generate embeddings for
            decoder_cond: Conditioning embeddings
            cache: Optional KV cache

        Returns:
            Dictionary with prepared inputs
        """
        # Embed speech tokens
        inputs_embeds = self.speech_emb(input_ids)

        # On first step, prepend conditioning
        if cache is None or len([c for c in cache if c is not None]) == 0:
            if decoder_cond.shape[0] != inputs_embeds.shape[0]:
                decoder_cond = mx.broadcast_to(
                    decoder_cond,
                    (
                        inputs_embeds.shape[0],
                        decoder_cond.shape[1],
                        decoder_cond.shape[2],
                    ),
                )
            inputs_embeds = mx.concatenate([decoder_cond, inputs_embeds], axis=1)

        return {
            "inputs_embeds": inputs_embeds,
            "cache": cache,
            "use_cache": True,
        }
