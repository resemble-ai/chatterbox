# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
MLX implementation of T3 (Token-to-Token) TTS model.
Converted from PyTorch version in t3.py
"""

import logging
import os
from typing import Optional, Dict, List
import mlx.core as mx
import mlx.nn as nn
from tqdm import tqdm

from .modules.llama_mlx import LlamaModelMLX, LlamaConfigMLX
from .modules.learned_pos_emb_mlx import LearnedPositionEmbeddingsMLX
from .modules.cond_enc_mlx import T3CondEncMLX, T3CondMLX
from .inference.t3_mlx_backend import T3MLXBackend
from ..t3.llama_configs import LLAMA_CONFIGS
from ..t3.modules.t3_config import T3Config
from ..utils import get_memory_info, is_debug

logger = logging.getLogger(__name__)


def _log_t3_memory(label: str):
    """Log memory for T3 MLX debugging. Enabled via DEBUG_MEMORY=1."""
    if not is_debug() and os.environ.get("DEBUG_MEMORY", "0") != "1":
        return
    info = get_memory_info()
    parts = [f"[T3_MLX] {label}:", f"Sys={info['sys_used_gb']:.2f}GB"]
    if "mps_allocated_mb" in info:
        parts.append(f"MPS={info['mps_allocated_mb']:.0f}MB")
    mx.eval(mx.array([0]))  # Force MLX sync
    print(" | ".join(parts))


def _ensure_BOT_EOT(text_tokens: mx.array, hp: T3Config):
    """Validate that text tokens contain required start/stop tokens."""
    B = text_tokens.shape[0]
    has_start = mx.sum(text_tokens == hp.start_text_token) >= B
    has_stop = mx.sum(text_tokens == hp.stop_text_token) >= B
    assert has_start, "missing start_text_token"
    assert has_stop, "missing stop_text_token"


class T3MLX(nn.Module):
    """
    MLX implementation of Token-To-Token (T3) TTS model.

    This is a complete port of the PyTorch T3 model to MLX, optimized for Apple Silicon.
    Uses Llama as the backbone transformer with custom embeddings and conditioning.
    """

    def __init__(self, hp: Optional[T3Config] = None):
        """
        Initialize T3 MLX model.

        Args:
            hp: T3Config hyperparameters. If None, uses English-only default.
        """
        super().__init__()

        if hp is None:
            hp = T3Config.english_only()

        self.hp = hp

        # Initialize Llama backbone (MLX version)
        llama_config_dict = LLAMA_CONFIGS[hp.llama_config_name]
        self.cfg = LlamaConfigMLX.from_dict(llama_config_dict)
        self.tfmr = LlamaModelMLX(self.cfg)
        self.dim = self.cfg.hidden_size

        # Conditioning encoder
        self.cond_enc = T3CondEncMLX(hp)

        # Token embeddings
        self.text_emb = nn.Embedding(hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(hp.speech_tokens_dict_size, self.dim)

        # Position embeddings
        if hp.input_pos_emb == "learned":
            max_text_seq_len = hp.max_text_tokens + 2
            self.text_pos_emb = LearnedPositionEmbeddingsMLX(max_text_seq_len, self.dim)

            max_mel_seq_len = hp.max_speech_tokens + 4
            self.speech_pos_emb = LearnedPositionEmbeddingsMLX(
                max_mel_seq_len, self.dim
            )

        # Logit projection heads
        self.text_head = nn.Linear(
            self.cfg.hidden_size, hp.text_tokens_dict_size, bias=False
        )
        self.speech_head = nn.Linear(
            self.cfg.hidden_size, hp.speech_tokens_dict_size, bias=False
        )

        # Backend for generation
        self.patched_model = None

    def prepare_conditioning(self, t3_cond: T3CondMLX) -> mx.array:
        """
        Prepare conditioning embeddings.

        Args:
            t3_cond: T3CondMLX object with conditioning data

        Returns:
            Conditioning embeddings of shape (B, len_cond, dim)
        """
        # Embed speech prompt tokens if provided
        if (
            t3_cond.cond_prompt_speech_tokens is not None
            and t3_cond.cond_prompt_speech_emb is None
        ):
            speech_tokens_emb = self.speech_emb(t3_cond.cond_prompt_speech_tokens)
            speech_pos_emb = self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
            t3_cond.cond_prompt_speech_emb = speech_tokens_emb + speech_pos_emb

        return self.cond_enc(t3_cond)  # (B, len_cond, dim)

    def prepare_input_embeds(
        self,
        *,
        t3_cond: T3CondMLX,
        text_tokens: mx.array,
        speech_tokens: mx.array,
        cfg_weight: float = 0.0,
    ) -> tuple:
        """
        Prepare input embeddings from conditioning, text, and speech tokens.

        Args:
            t3_cond: Conditioning data
            text_tokens: Text token IDs (B, L) - batch size is 2 when cfg_weight > 0
            speech_tokens: Speech token IDs
            cfg_weight: Classifier-free guidance weight

        Returns:
            Tuple of (embeddings, conditioning_length)
        """
        # Prepare conditioning embeddings
        cond_emb = self.prepare_conditioning(t3_cond)  # (B, len_cond, dim)

        # Text embeddings
        text_emb = self.text_emb(text_tokens)  # (B, len_text, dim)

        # CFG: zero out text embedding for unconditional branch BEFORE adding position embeddings
        # This matches PyTorch behavior where text_emb[1].zero_() is called before pos_emb addition
        # Note: text_tokens already has batch size 2 when CFG is enabled
        if cfg_weight > 0.0 and text_emb.shape[0] >= 2:
            # Zero out the second batch item (unconditional) - only the content embedding, not position
            text_emb_cond = text_emb[0:1]
            text_emb_uncond = mx.zeros_like(text_emb[1:2])
            text_emb = mx.concatenate([text_emb_cond, text_emb_uncond], axis=0)

        # Add text position embeddings AFTER zeroing out unconditional text
        if self.hp.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)

        # Speech embeddings
        speech_emb = self.speech_emb(speech_tokens)  # (B, len_speech, dim)

        # Add speech position embeddings
        if self.hp.input_pos_emb == "learned":
            speech_emb = speech_emb + self.speech_pos_emb(speech_tokens)

        # Broadcast embeddings to match batch sizes
        batch_size = text_emb.shape[0]

        # Expand conditioning if needed
        if cond_emb.shape[0] != batch_size:
            cond_emb = mx.broadcast_to(
                cond_emb, (batch_size, cond_emb.shape[1], cond_emb.shape[2])
            )

        # Expand speech embeddings if needed
        if speech_emb.shape[0] != batch_size:
            speech_emb = mx.broadcast_to(
                speech_emb, (batch_size, speech_emb.shape[1], speech_emb.shape[2])
            )

        len_cond = cond_emb.shape[1]

        # Concatenate: [conditioning, text, speech]
        embeds = mx.concatenate([cond_emb, text_emb, speech_emb], axis=1)

        return embeds, len_cond

    def __call__(
        self,
        *,
        t3_cond: T3CondMLX,
        text_tokens: mx.array,
        text_token_lens: mx.array,
        speech_tokens: mx.array,
        speech_token_lens: mx.array,
        training: bool = False,
    ) -> Dict:
        """
        Forward pass for training/evaluation.

        Args:
            t3_cond: Conditioning data
            text_tokens: Text token IDs (B, len_text)
            text_token_lens: Actual lengths of text sequences (B,)
            speech_tokens: Speech token IDs (B, len_speech)
            speech_token_lens: Actual lengths of speech sequences (B,)
            training: Whether in training mode

        Returns:
            Dictionary with logits and latents
        """
        _ensure_BOT_EOT(text_tokens, self.hp)

        # Prepare input embeddings
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
        )

        # Forward through transformer
        tfmr_out = self.tfmr(
            inputs_embeds=embeds,
            cache=None,  # No cache during training
            output_hidden_states=True,
        )
        hidden_states = tfmr_out["hidden_states"][-1]  # (B, seq, dim)

        # Extract text and speech latents
        len_text = text_tokens.shape[1]
        len_speech = speech_tokens.shape[1]

        text_latents = hidden_states[:, len_cond : len_cond + len_text, :]
        speech_latents = hidden_states[
            :, len_cond + len_text : len_cond + len_text + len_speech, :
        ]

        # Project to logits
        text_logits = self.text_head(text_latents)
        speech_logits = self.speech_head(speech_latents)

        return {
            "text_logits": text_logits,
            "text_latents": text_latents,
            "speech_logits": speech_logits,
            "speech_latents": speech_latents,
            "hidden_states": hidden_states,
        }

    def loss_fn(
        self,
        *,
        t3_cond: T3CondMLX,
        text_tokens: mx.array,
        text_token_lens: mx.array,
        speech_tokens: mx.array,
        speech_token_lens: mx.array,
    ) -> tuple:
        """
        Compute training losses.

        Args:
            t3_cond: Conditioning data
            text_tokens: Text token IDs
            text_token_lens: Text sequence lengths
            speech_tokens: Speech token IDs
            speech_token_lens: Speech sequence lengths

        Returns:
            Tuple of (text_loss, speech_loss)
        """
        text_tokens.shape[1]
        speech_tokens.shape[1]

        # Forward pass
        out = self(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            training=True,
        )

        # Compute masked cross-entropy losses
        def masked_cross_entropy(logits, targets, lengths):
            """Cross-entropy with length-based masking."""
            B, L = targets.shape

            # Create mask
            mask = mx.arange(L)[None, :] < lengths[:, None]  # (B, L)

            # Compute log probabilities
            log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)

            # Gather target log probs
            mx.arange(B)[:, None, None]  # (B, 1, 1)
            mx.arange(L)[None, :, None]  # (1, L, 1)
            mx.expand_dims(targets, -1)  # (B, L, 1)

            # Manual indexing since MLX doesn't have torch.gather
            target_log_probs = mx.zeros((B, L))
            for b in range(B):
                for pos in range(L):
                    target_log_probs[b, pos] = log_probs[b, pos, targets[b, pos]]

            # Apply mask and compute mean
            masked_log_probs = mx.where(mask, target_log_probs, 0.0)
            loss = -mx.sum(masked_log_probs) / mx.maximum(mx.sum(mask), 1.0)

            return loss

        loss_text = masked_cross_entropy(
            out["text_logits"], text_tokens, text_token_lens
        )
        loss_speech = masked_cross_entropy(
            out["speech_logits"], speech_tokens, speech_token_lens
        )

        return loss_text, loss_speech

    def generate(
        self,
        *,
        t3_cond: T3CondMLX,
        text_tokens: mx.array,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        repetition_penalty: float = 1.2,
        cfg_weight: float = 0.5,
        show_progress: bool = True,
        use_alignment_analyzer: bool = True,  # Enable by default for quality control
    ) -> mx.array:
        """
        Generate speech tokens autoregressively.

        Args:
            t3_cond: Conditioning data
            text_tokens: Text token IDs (1D or 2D)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            min_p: Minimum probability threshold
            repetition_penalty: Penalty for repeated tokens
            cfg_weight: Classifier-free guidance weight
            use_alignment_analyzer: Whether to use alignment analyzer for quality control

        Returns:
            Generated speech token IDs
        """
        # Import AlignmentStreamAnalyzerMLX
        from .inference.alignment_stream_analyzer_mlx import (
            AlignmentStreamAnalyzerMLX,
            LLAMA_ALIGNED_HEADS,
        )

        # Validate text tokens
        if text_tokens.ndim == 1:
            text_tokens = mx.expand_dims(text_tokens, 0)

        _ensure_BOT_EOT(text_tokens, self.hp)

        # Initialize generation
        if max_new_tokens is None:
            max_new_tokens = self.hp.max_speech_tokens

        # Set memory limits to prevent excessive memory usage
        # Cache limit of 0 forces immediate reclamation (more aggressive)
        # This helps prevent memory spikes during long generations
        mx.set_cache_limit(0)

        _log_t3_memory("generate_start")

        # Start with BOS token
        bos_token = mx.array([[self.hp.start_speech_token]])
        bos_embed = self.speech_emb(bos_token)
        bos_embed = bos_embed + self.speech_pos_emb.get_fixed_embedding(0)

        # Prepare conditioning
        initial_speech_tokens = bos_token
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech_tokens,
            cfg_weight=cfg_weight,
        )

        _log_t3_memory("after_prepare_embeds")

        # CFG: duplicate BOS embed to match embeds batch size if needed
        if cfg_weight > 0.0 and embeds.shape[0] == 2:
            bos_embed = mx.concatenate([bos_embed, bos_embed], axis=0)

        # Combine conditioning and BOS
        inputs_embeds = mx.concatenate([embeds, bos_embed], axis=1)

        # Initialize alignment analyzer for quality control
        alignment_analyzer = None
        if use_alignment_analyzer:
            len_text = text_tokens.shape[1]
            alignment_analyzer = AlignmentStreamAnalyzerMLX(
                text_tokens_slice=(len_cond, len_cond + len_text),
                eos_idx=self.hp.stop_speech_token,
            )
            logger.info("✓ AlignmentStreamAnalyzerMLX enabled for quality control")

        # Initialize backend if needed
        if self.patched_model is None:
            self.patched_model = T3MLXBackend(
                llama_model=self.tfmr,
                speech_emb=self.speech_emb,
                speech_head=self.speech_head,
                config=self.cfg,
                alignment_stream_analyzer=alignment_analyzer,
            )
        else:
            self.patched_model.reset_state()
            # Update alignment analyzer in backend
            self.patched_model.alignment_stream_analyzer = alignment_analyzer

        # Initial forward pass
        # Enable output_attentions if using alignment analyzer
        output_attentions = use_alignment_analyzer
        output = self.patched_model(
            inputs_embeds=inputs_embeds,
            cache=None,
            decoder_cond=None,
            use_cache=True,
            output_hidden_states=False,
            output_attentions=output_attentions,
        )

        cache = output["cache"]

        # Update alignment analyzer with initial attention weights
        if alignment_analyzer is not None and "attentions" in output:
            for idx, (layer_idx, head_idx) in enumerate(LLAMA_ALIGNED_HEADS):
                alignment_analyzer.update_attention(
                    output["attentions"],
                    buffer_idx=idx,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                )

        # Force immediate evaluation of initial forward pass to prevent lazy graph buildup
        if cache is not None and len(cache) > 0 and cache[0] is not None:
            mx.eval(cache[0][0])

        # Track generated tokens:
        # - generated_token_ids: Python ints for repetition penalty (memory efficient)
        # - generated_tokens: MLX arrays for final concatenation
        generated_token_ids: List[int] = []  # For repetition penalty
        generated_tokens: List[mx.array] = []  # For final result

        # Import sampling utilities
        from .inference.sampling_utils_mlx import (
            apply_repetition_penalty,
            apply_top_p,
            apply_min_p,
        )

        # Generation loop
        token_iterator = range(max_new_tokens)
        if show_progress:
            token_iterator = tqdm(token_iterator, desc="Generating", dynamic_ncols=True)
        for i in token_iterator:
            logits_step = output["logits"][:, -1, :]  # (B, vocab)

            # CFG: combine conditional and unconditional predictions
            if cfg_weight > 0.0:
                cond = logits_step[0:1, :]
                uncond = logits_step[1:2, :]
                logits = cond + cfg_weight * (cond - uncond)
            else:
                logits = logits_step[0:1, :]

            # CRITICAL: Apply alignment analyzer BEFORE other sampling modifications
            # This allows the analyzer to force/suppress EOS based on alignment
            if alignment_analyzer is not None:
                # Pass the last generated token for repetition tracking
                last_token = (
                    generated_token_ids[-1] if len(generated_token_ids) > 0 else None
                )
                logits = alignment_analyzer.step(logits, next_token=last_token)

            # Apply repetition penalty (uses Python ints, no MLX array concatenation)
            logits = apply_repetition_penalty(
                logits, generated_token_ids, repetition_penalty
            )

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply sampling filters
            logits = apply_min_p(logits, min_p)
            logits = apply_top_p(logits, top_p)

            # Sample next token
            probs = mx.softmax(logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
            next_token = mx.reshape(next_token, (1, 1))

            # int() implicitly evaluates
            token_val = int(next_token[0, 0])

            # Store token value (Python int) and tensor separately
            generated_token_ids.append(token_val)
            generated_tokens.append(next_token)

            # Check for EOS
            if token_val == self.hp.stop_speech_token:
                logger.info(f"✅ EOS token detected at step {i+1}")
                break

            # Get embedding for next token
            next_embed = self.speech_emb(next_token)
            next_embed = next_embed + self.speech_pos_emb.get_fixed_embedding(i + 1)

            # CFG: duplicate embedding to match cache batch size
            if cfg_weight > 0.0 and cache is not None and len(cache) > 0:
                # Check if cache expects batch size of 2
                if cache[0] is not None and cache[0][0].shape[0] == 2:
                    next_embed = mx.concatenate([next_embed, next_embed], axis=0)

            # Forward with cache
            output = self.patched_model(
                inputs_embeds=next_embed,
                cache=cache,
                use_cache=True,
                output_hidden_states=False,
                output_attentions=output_attentions,
            )

            cache = output["cache"]

            # Update alignment analyzer with new attention weights
            if alignment_analyzer is not None and "attentions" in output:
                for idx, (layer_idx, head_idx) in enumerate(LLAMA_ALIGNED_HEADS):
                    alignment_analyzer.update_attention(
                        output["attentions"],
                        buffer_idx=idx,
                        layer_idx=layer_idx,
                        head_idx=head_idx,
                    )

            # More aggressive memory management to prevent spikes
            # Eval every 25 steps to bound lazy evaluation graph
            if i > 0 and i % 25 == 0:
                if cache is not None and len(cache) > 0 and cache[0] is not None:
                    mx.eval(cache[0][0])
                # Clear MLX cache periodically to release intermediate computations
                mx.clear_cache()

        _log_t3_memory("generate_complete")

        # Concatenate generated tokens
        if len(generated_tokens) > 0:
            result = mx.concatenate(generated_tokens, axis=1)
        else:
            result = mx.array([[]], dtype=mx.int32)

        # Force evaluation before clearing references
        mx.eval(result)

        # Clear references to help GC
        del generated_tokens
        del generated_token_ids
        del cache
        del output

        # Clear MLX memory cache to prevent accumulation across generations
        mx.clear_cache()

        import gc

        gc.collect()

        return result

    def inference(
        self,
        *,
        t3_cond: T3CondMLX,
        text_tokens: mx.array,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        repetition_penalty: float = 1.2,
        cfg_weight: float = 0.5,
        show_progress: bool = True,
        use_alignment_analyzer: bool = True,  # Enable alignment analyzer by default
        # PyTorch API compatibility - ignored for MLX
        initial_speech_tokens: Optional[mx.array] = None,
        prepend_prompt_speech_tokens: Optional[mx.array] = None,
        num_return_sequences: int = 1,
        stop_on_eos: bool = True,
        do_sample: bool = True,
        length_penalty: float = 1.0,
    ) -> mx.array:
        """
        Inference wrapper for compatibility with PyTorch T3 API.

        This is an alias for generate() with additional parameters for PyTorch API compatibility.
        Some parameters are ignored in the MLX implementation.

        Args:
            t3_cond: Conditioning data
            text_tokens: Text token IDs (1D or 2D)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            min_p: Minimum probability threshold
            repetition_penalty: Penalty for repeated tokens
            cfg_weight: Classifier-free guidance weight
            use_alignment_analyzer: Whether to use alignment analyzer for quality control
            initial_speech_tokens: Ignored (for PyTorch API compatibility)
            prepend_prompt_speech_tokens: Ignored (for PyTorch API compatibility)
            num_return_sequences: Ignored (for PyTorch API compatibility)
            stop_on_eos: Ignored (for PyTorch API compatibility)
            do_sample: Ignored (for PyTorch API compatibility)
            length_penalty: Ignored (for PyTorch API compatibility)

        Returns:
            Generated speech token IDs
        """
        return self.generate(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            cfg_weight=cfg_weight,
            show_progress=show_progress,
            use_alignment_analyzer=use_alignment_analyzer,
        )
