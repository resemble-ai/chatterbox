#chatterbox/src/chatterbox/models/t3/t3.py

# Copyright (c) 2025 Resemble AI
# MIT License
import logging
from typing import Union, Optional, List

logger = logging.getLogger(__name__)

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import LlamaModel, LlamaConfig
from transformers.generation.logits_process import TopPLogitsWarper, RepetitionPenaltyLogitsProcessor, MinPLogitsWarper

from .modules.learned_pos_emb import LearnedPositionEmbeddings

from .modules.cond_enc import T3CondEnc, T3Cond
from .modules.t3_config import T3Config
from .llama_configs import LLAMA_CONFIGS
from .inference.t3_hf_backend import T3HuggingfaceBackend
from .inference.alignment_stream_analyzer import AlignmentStreamAnalyzer
from ..utils import AttrDict


logger = logging.getLogger(__name__)


def _ensure_BOT_EOT(text_tokens: Tensor, hp):
    B = text_tokens.size(0)
    assert (text_tokens == hp.start_text_token).int().sum() >= B, "missing start_text_token"
    assert (text_tokens == hp.stop_text_token).int().sum() >= B, "missing stop_text_token"


class T3(nn.Module):
    """
    Token-To-Token (T3) TTS model using huggingface transformer models as backbones,
        * tokenization, including start / stop tokens are always added externally to this class
        * conditioning data like CLAP, emotion, etc are all in a separate file for more modularity
        * careful! this class assumes relative positional encoding -- with absolute PE, we would at
            least want to reset the position to 0 when speech tokens begin, and optionally use a
            different PE embedding space for speech.
    """

    def __init__(self, hp=None, dtype=torch.float32):
        if hp is None:
            hp = T3Config.english_only()  # Default to English-only config for backward compatibility
        super().__init__()
        self.hp = hp
        
        config_dict = LLAMA_CONFIGS[hp.llama_config_name].copy()
        # Note: We no longer need to set torch_dtype in the config, as it's not used for scratch initialization.
        # config_dict['torch_dtype'] = dtype 
        self.cfg = LlamaConfig(**config_dict)

        # --- DEFINITIVE FIX for Flash Attention Initialization ---
        # The LlamaModel constructor does not respect config.torch_dtype for scratch initialization;
        # it uses the global default. We temporarily set the global default dtype to our target dtype
        # to ensure the model's layers are created in BF16 from the start.
        original_dtype = torch.get_default_dtype()
        try:
            if dtype in [torch.float16, torch.bfloat16]:
                torch.set_default_dtype(dtype)
            
            self.tfmr = LlamaModel(self.cfg)

        finally:
            # Always restore the original default dtype to avoid side-effects elsewhere.
            torch.set_default_dtype(original_dtype)
        # --- End of Fix ---

        self.dim = self.cfg.hidden_size
        self.deepspeed_patch_applied = False

        # conditioning / embedding
        self.cond_enc = T3CondEnc(hp)
        self.text_emb = nn.Embedding(hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(hp.speech_tokens_dict_size, self.dim)

        # custom position embedding
        if hp.input_pos_emb == "learned":
            max_text_seq_len = hp.max_text_tokens + 2
            self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, self.dim)

            max_mel_seq_len = hp.max_speech_tokens + 2 + 2
            self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

        # logit projection
        self.text_head = nn.Linear(self.cfg.hidden_size, hp.text_tokens_dict_size, bias=False)
        self.speech_head = nn.Linear(self.cfg.hidden_size, hp.speech_tokens_dict_size, bias=False)
        self.compiled = False

    @property
    def device(self):
        return self.speech_head.weight.device

    def prepare_conditioning(self, t3_cond: T3Cond):
        """
        Token cond data needs to be embedded, so that needs to be here instead of in `T3CondEnc`.
        """
        if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
            t3_cond.cond_prompt_speech_emb = self.speech_emb(t3_cond.cond_prompt_speech_tokens) + \
                self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
        return self.cond_enc(t3_cond)  # (B, len_cond, dim)

    def prepare_input_embeds(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        cfg_weight: float = 0.0,
    ):
        # prepare input embeddings (skip backbone tranformer embeddings)
        cond_emb = self.prepare_conditioning(t3_cond)  # (B, len_cond, dim)
        text_emb = self.text_emb(text_tokens)  # (B, len_text, dim)
        if cfg_weight > 0.0:
            B = text_tokens.size(0) // 2
            text_emb[B:].zero_()  # CFG uncond

        speech_emb = self.speech_emb(speech_tokens)  # (B, len_speech, dim)
        if self.hp.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)
            speech_emb = speech_emb + self.speech_pos_emb(speech_tokens)
        len_cond = cond_emb.size(1)

        if cond_emb.size(0) != text_emb.size(0):
             cond_emb = cond_emb.expand(text_emb.size(0), -1, -1)

        # concat
        embeds = torch.cat([cond_emb, text_emb, speech_emb], dim=1) # (B, length, dim)
        return embeds, len_cond

    def forward(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        speech_token_lens: torch.LongTensor,
        training=False,
    ):
        _ensure_BOT_EOT(text_tokens, self.hp)

        # prepare custom input embeds
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
        )

        # backbone tranformer forward
        tfmr_out = self.tfmr.forward(
            input_ids=None,
            # position_ids=position_ids, # TODO? ROPE should be fine?
            inputs_embeds=embeds,
            output_hidden_states=True,
            return_dict=True,
            use_cache=(not training),
        )
        hidden_states = tfmr_out.hidden_states[-1]  # final tfmr layer output, (B, seq, dim)

        # post-processing: splice out text and speech parts of hidden states
        len_text = text_tokens.size(1)
        len_speech = speech_tokens.size(1)
        B, _, dim = hidden_states.shape
        device, dtype = hidden_states.device, hidden_states.dtype
        text_latents = torch.zeros(B, len_text, dim, dtype=dtype, device=device)
        speech_latents = torch.zeros(B, len_speech, dim, dtype=dtype, device=device)
        ttl, stl = text_token_lens, speech_token_lens
        for i in range(B):
            text_end = len_cond + ttl[i].item()
            speech_start = len_cond + text_tokens.size(1)
            speech_end = speech_start + stl[i].item()
            text_latents[i, :ttl[i]] = hidden_states[i, len_cond:text_end]
            speech_latents[i, :stl[i]] = hidden_states[i, speech_start:speech_end]

        # logit projection
        text_logits = self.text_head(text_latents)
        speech_logits = self.speech_head(speech_latents)

        return AttrDict(
            text_logits=text_logits,
            text_latents=text_latents,
            speech_logits=speech_logits,
            speech_latents=speech_latents,
            hidden_states=hidden_states,
        )

    def loss(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        speech_token_lens: torch.LongTensor,
    ):
        "training method"
        len_text = text_tokens.size(1)
        len_speech = speech_tokens.size(1)
        assert len_text == text_token_lens.max()
        assert len_speech == speech_token_lens.max()

        out = self.forward(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            training=True,
        )  # (B, seq, vocab_size)

        # Calc CCE losses
        IGNORE_ID = -100
        device = out.text_logits.device
        mask_text = torch.arange(len_text, device=device)[None] >= text_token_lens[:, None]  # (B, len_text)
        mask_speech = torch.arange(len_speech, device=device)[None] >= speech_token_lens[:, None]  # (B, len_speech)
        masked_text = text_tokens.masked_fill(mask_text, IGNORE_ID)
        masked_speech = speech_tokens.masked_fill(mask_speech, IGNORE_ID)
        loss_text = F.cross_entropy(out.text_logits, masked_text, ignore_index=IGNORE_ID)
        loss_speech = F.cross_entropy(out.speech_logits, masked_speech, ignore_index=IGNORE_ID)

        return loss_text, loss_speech


    @torch.inference_mode()
    def inference(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: Tensor,
        initial_speech_tokens: Optional[Tensor]=None,
        # misc conditioning
        prepend_prompt_speech_tokens: Optional[Tensor]=None,
        # HF generate args
        num_return_sequences=1,
        max_new_tokens=None,
        stop_on_eos=True,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        min_p=0.05,
        length_penalty=1.0,
        repetition_penalty=1.2,
        cfg_weight=0.5,
    ):
        """
        Optimized batched inference with CFG support and early stopping.
        """
        # Validate / sanitize inputs
        assert prepend_prompt_speech_tokens is None, "not implemented"
        _ensure_BOT_EOT(text_tokens, self.hp)
        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)
        
        # Determine original batch size (B_orig) before CFG duplication
        B_total = text_tokens.size(0)
        if cfg_weight > 0.0:
            # Inputs are expected to be already duplicated if CFG is used (handled in ChatterboxTTS.generate)
            if B_total % 2 != 0:
                 raise ValueError("Batch size must be even when using CFG due to input duplication.")
            B_orig = B_total // 2
        else:
            B_orig = B_total

        max_new_tokens = max_new_tokens or self.hp.max_speech_tokens

        # Default initial speech to a single start-of-speech token
        if initial_speech_tokens is None:
            # Use B_total here as initial_speech_tokens must cover the full batch (cond + uncond)
            initial_speech_tokens = self.hp.start_speech_token * torch.ones((B_total, 1), dtype=torch.long, device=self.device)
        
        len_initial_speech = initial_speech_tokens.size(1)

        # Prepare custom input embeds [Cond, Text, InitialSpeech]
        # This correctly prepares the embeddings including positional information for the prefill phase.
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech_tokens,
            cfg_weight=cfg_weight,
        )

        # Initialize the Huggingface backend if not already done
        if not self.compiled:
            # Default to None for English models, only create for multilingual
            alignment_stream_analyzer = None
            if self.hp.is_multilingual:
                # NOTE: Ensure AlignmentStreamAnalyzer supports batching if required.
                logger.info("Initializing AlignmentStreamAnalyzer for multilingual support.")
                
                # Assuming AlignmentStreamAnalyzer is imported correctly in the file scope
                try:
                    alignment_stream_analyzer = AlignmentStreamAnalyzer(
                        self.tfmr,
                        None,
                        text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
                        alignment_layer_idx=9,
                        eos_idx=self.hp.stop_speech_token,
                    )
                except Exception as e:
                     logger.warning(f"AlignmentStreamAnalyzer initialization failed: {e}. Multilingual stability might be affected.")


            patched_model = T3HuggingfaceBackend(
                config=self.cfg,
                llama=self.tfmr,
                speech_enc=self.speech_emb,
                speech_head=self.speech_head,
                alignment_stream_analyzer=alignment_stream_analyzer,
            )
            self.patched_model = patched_model
            self.compiled = True

        inputs_embeds = embeds
        device = embeds.device

        # ---- Initial Forward Pass (Prefill/Prompt Processing) ----
        output = self.patched_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        # Initialize kv_cache with the full context.
        past = output.past_key_values

        # ---- Generation Loop (Decoding) using kv_cache ----

        # Instantiate the logits processors.
        top_p_warper = TopPLogitsWarper(top_p=top_p)
        min_p_warper = MinPLogitsWarper(min_p=min_p)
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))

        # Pre-allocate tensor for generated IDs (only the conditional part, B_orig)
        # We use the stop token as the padding value.
        max_len = len_initial_speech + max_new_tokens
        generated_ids_cond = torch.full((B_orig, max_len), self.hp.stop_speech_token, dtype=torch.long, device=device)
        # Copy the initial tokens (e.g., BOS)
        generated_ids_cond[:, :len_initial_speech] = initial_speech_tokens[:B_orig, :]

        # Track which sequences are finished
        is_finished = torch.zeros(B_orig, dtype=torch.bool, device=device)
        current_token_idx = len_initial_speech

        for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True):
            # Get the logits for the last time step
            logits_step = output.logits[:, -1, :] # (B_total, V)
                            
            # --- CFG combination ---
            if cfg_weight > 0.0:
                # Split the logits (B_total, V) into conditional and unconditional parts
                cond, uncond = torch.split(logits_step, B_orig, dim=0)
                cfg = torch.as_tensor(cfg_weight, device=cond.device, dtype=cond.dtype)
                # Combine using CFG formula
                logits = cond + cfg * (cond - uncond) # (B_orig, V)
            else:
                logits = logits_step # (B_orig, V)
            
            
            # --- Apply Logit Processors and Sampling ---

            # Apply alignment stream analyzer (if present, assumes it handles batching)
            if self.patched_model.alignment_stream_analyzer is not None:
                 # We omit passing the last token here; the analyzer must rely on internal state if needed.
                 # This might need refinement for robust multilingual batched generation.
                 logits = self.patched_model.alignment_stream_analyzer.step(logits, next_token=None)

            # Prepare inputs for processors: generated IDs up to the current step
            ids_for_proc = generated_ids_cond[:, :current_token_idx]

            # Apply repetition penalty
            logits = repetition_penalty_processor(ids_for_proc, logits)  # expects (B_orig, V)
            
            # Apply temperature scaling.
            if temperature != 1.0:
                logits = logits / temperature
                
            # Apply min_p and top_p filtering
            logits = min_p_warper(ids_for_proc, logits)
            logits = top_p_warper(ids_for_proc, logits)

            # Convert logits to probabilities and sample the next token.
            # Use FP32 for softmax for better numerical stability when using half-precision.
            probs = torch.softmax(logits.float(), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # shape: (B_orig, 1)

            # --- Update Generation State and Check Stopping Condition ---

            # If a sequence is finished, force the generation of the EOS token (padding)
            next_token = torch.where(
                is_finished.unsqueeze(1),
                torch.tensor(self.hp.stop_speech_token, device=device, dtype=torch.long),
                next_token
            )

            # Update the pre-allocated tensor in-place
            generated_ids_cond[:, current_token_idx] = next_token.squeeze(-1)
            current_token_idx += 1

            # Update finished status (check if the newly generated token is EOS)
            is_finished = is_finished | (next_token.squeeze(-1) == self.hp.stop_speech_token)

            # Check if all sequences are finished (Early Stopping)
            if is_finished.all():
                logger.info(f"✅ All sequences finished. Stopping generation at step {i+1}")
                break

            # --- Prepare for Next Step ---

            # Get embedding for the new token.
            next_token_embed = self.speech_emb(next_token)
            
            # Apply positional embedding for the next step
            if self.hp.input_pos_emb == "learned":
                 # The position index is len_initial_speech + i
                next_token_embed = next_token_embed + self.speech_pos_emb.get_fixed_embedding(len_initial_speech + i)

            # For CFG, duplicate the embeddings for the full batch (B_total)
            if cfg_weight > 0.0:
                # Unconditional part uses the same embeddings as the conditional part for the next step input
                next_token_embed = torch.cat([next_token_embed, next_token_embed], dim=0)

            # Forward pass with only the new token and the cached past.
            output = self.patched_model(
                inputs_embeds=next_token_embed,
                past_key_values=past,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            # Update the kv_cache.
            past = output.past_key_values

        # Un-batch and trim EOS token
        output_tokens = []
        for gen_ids in generated_ids_cond:
            # Remove initial speech tokens that were passed in
            gen_ids = gen_ids[len_initial_speech:]
            # Find the first EOS token and trim
            eos_idx = (gen_ids == self.hp.stop_speech_token).nonzero(as_tuple=True)[0]
            if len(eos_idx) > 0:
                gen_ids = gen_ids[:eos_idx[0]]
            output_tokens.append(gen_ids)

        return output_tokens