# Copyright (c) 2025 Resemble AI
# MIT License
import logging
from typing import Union, Optional, List

logger = logging.getLogger(__name__)

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import LlamaModel, LlamaConfig, GPT2Config, GPT2Model
from .inference.kv_truncating_cache import KVTruncatingStaticCache
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    MinPLogitsWarper,
)
from .fast_min_p_warper import FastMinPLogitsWarper
from .fast_top_p_warper import FastTopPLogitsWarper
from .modules.learned_pos_emb import LearnedPositionEmbeddings

from .modules.cond_enc import T3CondEnc, T3Cond
from .modules.t3_config import T3Config
from .llama_configs import LLAMA_CONFIGS
from .inference.t3_hf_backend import T3HuggingfaceBackend
from .inference.alignment_stream_analyzer import AlignmentStreamAnalyzer
from .t3_cuda_graphs import T3StepCUDAGraphWrapper, get_next_bucket, TOKEN_LIMIT
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

    def __init__(self, hp=None):
        if hp is None:
            hp = T3Config.english_only()
        super().__init__()
        self.hp = hp

        config_dict = LLAMA_CONFIGS[hp.llama_config_name]
        self.is_gpt = config_dict.get("model_type") == "gpt2"

        if self.is_gpt:
            self.cfg = GPT2Config(**config_dict)
            self.tfmr = GPT2Model(self.cfg)
        else:
            self.cfg = LlamaConfig(**config_dict)
            self.tfmr = LlamaModel(self.cfg)

        self.dim = self.cfg.hidden_size
        self.deepspeed_patch_applied = False

        # conditioning / embedding
        self.cond_enc = T3CondEnc(hp)
        self.text_emb = nn.Embedding(hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(hp.speech_tokens_dict_size, self.dim)

        # custom position embedding
        self.text_pos_emb = None
        self.speech_pos_emb = None
        if hp.input_pos_emb == "learned":
            max_text_seq_len = hp.max_text_tokens + 2
            self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, self.dim)

            max_mel_seq_len = hp.max_speech_tokens + 2 + 2
            self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

        # logit projection
        self.text_head = nn.Linear(self.cfg.hidden_size, hp.text_tokens_dict_size, bias=False)
        self.speech_head = nn.Linear(self.cfg.hidden_size, hp.speech_tokens_dict_size, bias=self.is_gpt)
        self.compiled = False
        self.init_processors()

    @property
    def device(self):
        return self.speech_head.weight.device

    def init_processors(self, top_p=1.0, min_p=0.05, repetition_penalty=1.2):
        self.top_p_warper = FastTopPLogitsWarper(top_p=top_p, device=self.device)
        self.min_p_warper = FastMinPLogitsWarper(min_p=min_p, device=self.device)
        self.repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
        # Track as Python floats to avoid GPU tensor comparisons (which cause CUDA syncs)
        self._top_p = float(top_p)
        self._min_p = float(min_p)
        self._repetition_penalty = float(repetition_penalty)

    def update_processors(self, top_p, min_p, repetition_penalty):
        if self._top_p != top_p:
            self.top_p_warper.top_p = torch.tensor(top_p, device=self.top_p_warper.top_p.device)
            self._top_p = float(top_p)
        if self._min_p != min_p:
            self.min_p_warper.min_p = torch.tensor(min_p, device=self.min_p_warper.min_p.device)
            self._min_p = float(min_p)
        if self._repetition_penalty != repetition_penalty:
            self.repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
            self._repetition_penalty = float(repetition_penalty)

    def get_speech_pos_embedding_cache(self, max_gen_tokens, dtype):
        if not hasattr(self, '_speech_pos_embedding_cache') or \
                self._speech_pos_embedding_cache.size(0) < max_gen_tokens:
            cache = [self.speech_pos_emb.get_fixed_embedding(i) for i in range(max_gen_tokens)]
            self._speech_pos_embedding_cache = torch.stack(cache).to(device=self.device)
        return self._speech_pos_embedding_cache.to(dtype=dtype)

    def init_speech_embedding_cache(self, vocab_size, dtype):
        if not hasattr(self, '_speech_embedding_cache') or \
                self._speech_embedding_cache.size(0) < vocab_size:
            tokens = torch.arange(vocab_size, device=self.device)
            self._speech_embedding_cache = self.speech_emb(tokens)
        return self._speech_embedding_cache.to(dtype=dtype)

    def get_or_reset_static_cache(self, max_batch_size, max_cache_len):
        """Allocate StaticCache once; reset in-place on subsequent calls.
        CUDA graphs are captured with specific memory addresses — reusing the same
        object keeps those addresses stable across generation calls."""
        if not hasattr(self, '_static_kv_cache'):
            self._static_kv_cache = KVTruncatingStaticCache(
                config=self._patched_model_fast.config,
                max_batch_size=max_batch_size,
                max_cache_len=max_cache_len,
                device=self.device,
                dtype=self._patched_model_fast.dtype if hasattr(self._patched_model_fast, 'dtype') else torch.float32,
            )
            self._static_kv_cache_params = (max_batch_size, max_cache_len)
        elif self._static_kv_cache_params != (max_batch_size, max_cache_len):
            # Params changed — must reallocate (also invalidates captured graphs)
            if hasattr(self, '_cudagraph_wrapper'):
                self._cudagraph_wrapper.reset()
            self._static_kv_cache = KVTruncatingStaticCache(
                config=self._patched_model_fast.config,
                max_batch_size=max_batch_size,
                max_cache_len=max_cache_len,
                device=self.device,
                dtype=self._patched_model_fast.dtype if hasattr(self._patched_model_fast, 'dtype') else torch.float32,
            )
            self._static_kv_cache_params = (max_batch_size, max_cache_len)
        else:
            self._static_kv_cache.reset()
        return self._static_kv_cache

    def prepare_conditioning(self, t3_cond: T3Cond):
        """
        Token cond data needs to be embedded, so that needs to be here instead of in `T3CondEnc`.
        """
        if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
            t3_cond.cond_prompt_speech_emb = self.speech_emb(t3_cond.cond_prompt_speech_tokens)
            if not self.is_gpt:
                t3_cond.cond_prompt_speech_emb += self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
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
        if cfg_weight > 0.0 and not self.is_gpt:
            text_emb[1].zero_()  # CFG uncond

        speech_emb = self.speech_emb(speech_tokens)  # (B, len_speech, dim)
        if self.hp.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)
            speech_emb = speech_emb + self.speech_pos_emb(speech_tokens)
        len_cond = cond_emb.size(1)

        if cond_emb.size(0) != text_emb.size(0):
             cond_emb = cond_emb.expand(text_emb.size(0), -1, -1)

        # concat
        embeds = torch.stack([
            torch.cat((ce, te, se))
            for ce, te, se in zip(cond_emb, text_emb, speech_emb)
        ])  # (B, length, dim)
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
    
    def loss_new(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,        # (B, S_text_padded), includes BOS & EOS
        text_token_lens: torch.LongTensor,    # (B,), actual lengths including BOS & EOS
        speech_tokens: torch.LongTensor,      # (B, S_speech_padded), includes BOS & EOS
        speech_token_lens: torch.LongTensor,  # (B,), actual lengths including BOS & EOS
        labels_text: torch.LongTensor,        # (B, S_text_padded-1), already masked with –100
        labels_speech: torch.LongTensor       # (B, S_speech_padded-1), already masked with –100
    ):
        """
        Compute text and speech cross-entropy using pre-masked labels from the collator.
        Assumes:
        - labels_text[t] corresponds to predicting text_tokens[:, 1:] with –100 where ignored
        - labels_speech[t] corresponds to predicting speech_tokens[:, 1:] with –100 where ignored
        """

        # 1) Run model to get logits
        out = self.forward(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            training=True,
        )
        # out.text_logits: (B, S_text_padded, V_text)
        # out.speech_logits: (B, S_speech_padded, V_speech)
        device = out.text_logits.device
        IGNORE_ID = -100

        # --- Text Loss (use labels_text directly) ---
        # Align logits: predict t₁..EOS from inputs [BOS, t₁..]
        logits_for_text = out.text_logits[:, :-1, :].contiguous()  # (B, S_text_padded-1, V_text)
        # labels_text already has shape (B, S_text_padded-1) with –100 where masked
        if logits_for_text.size(1) == 0:
            loss_text = torch.tensor(0.0, device=device, requires_grad=self.training)
        else:
            loss_text = F.cross_entropy(
                logits_for_text.transpose(1, 2),  # (B, V_text, S_text_padded-1)
                labels_text,                      # (B, S_text_padded-1), ignore_index=–100
                ignore_index=IGNORE_ID
            )

        # --- Speech Loss (use labels_speech directly) ---
        logits_for_speech = out.speech_logits[:, :-1, :].contiguous()  # (B, S_speech_padded-1, V_speech)
        # labels_speech already has shape (B, S_speech_padded-1) with –100 where masked
        if logits_for_speech.size(1) == 0:
            loss_speech = torch.tensor(0.0, device=device, requires_grad=self.training)
        else:
            loss_speech = F.cross_entropy(
                logits_for_speech.transpose(1, 2),  # (B, V_speech, S_speech_padded-1)
                labels_speech,                      # (B, S_speech_padded-1), ignore_index=–100
                ignore_index=IGNORE_ID
            )

        return loss_text, loss_speech, out.speech_logits

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
        Args:
            text_tokens: a 1D (unbatched) or 2D (batched) tensor.
        """
        # Validate / sanitize inputs
        assert prepend_prompt_speech_tokens is None, "not implemented"
        _ensure_BOT_EOT(text_tokens, self.hp)
        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)

        # Default initial speech to a single start-of-speech token
        if initial_speech_tokens is None:
            initial_speech_tokens = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])

        # Prepare custom input embeds
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech_tokens,
            cfg_weight=cfg_weight,
        )

        # In order to use the standard HF generate method, we need to extend some methods to inject our custom logic
        # Note the llama-specific logic. Other tfmr types can be added later.

        self.compiled = False

        # TODO? synchronize the expensive compile function
        # with self.compile_lock:
        if not self.compiled:
            # Default to None for English models, only create for multilingual
            alignment_stream_analyzer = None
            if self.hp.is_multilingual:
                alignment_stream_analyzer = AlignmentStreamAnalyzer(
                    self.tfmr,
                    None,
                    text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
                    alignment_layer_idx=9, # TODO: hparam or something?
                    eos_idx=self.hp.stop_speech_token,
                )
                assert alignment_stream_analyzer.eos_idx == self.hp.stop_speech_token

            patched_model = T3HuggingfaceBackend(
                config=self.cfg,
                llama=self.tfmr,
                speech_enc=self.speech_emb,
                speech_head=self.speech_head,
                alignment_stream_analyzer=alignment_stream_analyzer,
            )
            self.patched_model = patched_model
            self.compiled = True

        # # Run normal generate method, which calls our custom extended methods
        # return self.patched_model.generate(
        #     inputs=initial_speech_tokens,
        #     decoder_cond=embeds,
        #     bos_token_id=self.hp.start_speech_token,
        #     eos_token_id=(self.hp.stop_speech_token if stop_on_eos else -1),
        #     pad_token_id=self.hp.stop_speech_token,
        #     max_new_tokens=max_new_tokens or self.hp.max_speech_tokens,
        #     num_return_sequences=num_return_sequences,
        #     temperature=temperature,
        #     min_p=min_p,
        #     length_penalty=length_penalty,
        #     repetition_penalty=repetition_penalty,
        #     do_sample=do_sample,
        #     # cache_implementation=None if not self.compiled else "static",
        # )

        device = embeds.device

        bos_token = torch.tensor([[self.hp.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = self.speech_emb(bos_token)  # shape: (B, 1, embed_dim)
        bos_embed = bos_embed + self.speech_pos_emb.get_fixed_embedding(0)

        # batch_size=2 for CFG
        bos_embed = torch.cat([bos_embed, bos_embed])

        # Combine condition and BOS token for the initial input
        inputs_embeds = torch.cat([embeds, bos_embed], dim=1)

        # Track generated token ids; start with the BOS token.
        generated_ids = bos_token.clone()
        predicted = []  # To store the predicted tokens

        # Update pre-instantiated logits processors with current call's parameters.
        self.update_processors(top_p=top_p, min_p=min_p, repetition_penalty=float(repetition_penalty))

        # ---- Initial Forward Pass (no kv_cache yet) ----
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

        # ---- Generation Loop using kv_cache ----
        for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True):
            logits_step = output.logits[:, -1, :]
            # CFG combine  → (1, V)
            cond   = logits_step[0:1, :]
            uncond = logits_step[1:2, :]
            cfg = torch.as_tensor(cfg_weight, device=cond.device, dtype=cond.dtype)
            logits = cond + cfg * (cond - uncond)
            
            # Apply alignment stream analyzer integrity checks
            if self.patched_model.alignment_stream_analyzer is not None:
                if logits.dim() == 1:            # guard in case something upstream squeezed
                    logits = logits.unsqueeze(0) # (1, V)
                # Pass the last generated token for repetition tracking
                last_token = generated_ids[0, -1].item() if len(generated_ids[0]) > 0 else None
                logits = self.patched_model.alignment_stream_analyzer.step(logits, next_token=last_token)  # (1, V)

            # Apply repetition penalty
            ids_for_proc = generated_ids[:1, ...]   # batch = 1
            logits = self.repetition_penalty_processor(ids_for_proc, logits)  # expects (B,V)
            
            # Apply temperature scaling.
            if temperature != 1.0:
                logits = logits / temperature
                
            # Apply min_p and top_p filtering
            logits = self.min_p_warper(ids_for_proc, logits)
            logits = self.top_p_warper(ids_for_proc, logits)

            # Convert logits to probabilities and sample the next token.
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # shape: (B, 1)

            predicted.append(next_token)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Check for EOS token.
            if next_token.view(-1) == self.hp.stop_speech_token:
                logger.info(f"✅ EOS token detected! Stopping generation at step {i+1}")
                break

            # Get embedding for the new token.
            next_token_embed = self.speech_emb(next_token)
            next_token_embed = next_token_embed + self.speech_pos_emb.get_fixed_embedding(i + 1)

            #  For CFG
            next_token_embed = torch.cat([next_token_embed, next_token_embed])

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

        # Concatenate all predicted tokens along the sequence dimension.
        predicted_tokens = torch.cat(predicted, dim=1)  # shape: (B, num_tokens)
        return predicted_tokens

    @torch.inference_mode()
    def inference_turbo(self, t3_cond, text_tokens, temperature=0.8, top_k=1000, top_p=0.95, repetition_penalty=1.2,
                        max_gen_len=1000):

        logits_processors = LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            logits_processors.append(TemperatureLogitsWarper(temperature))
        if top_k > 0:
            logits_processors.append(TopKLogitsWarper(top_k))
        if top_p < 1.0:
            logits_processors.append(TopPLogitsWarper(top_p))
        if repetition_penalty != 1.0:
            logits_processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))


        speech_start_token = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _ = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_start_token,
            cfg_weight=0.0,
        )

        generated_speech_tokens = []

        llm_outputs = self.tfmr(
            inputs_embeds=embeds,
            use_cache=True
        )

        hidden_states = llm_outputs[0]
        past_key_values = llm_outputs.past_key_values

        speech_hidden = hidden_states[:, -1:]
        speech_logits = self.speech_head(speech_hidden)

        processed_logits = logits_processors(speech_start_token, speech_logits[:, -1, :])
        probs = F.softmax(processed_logits, dim=-1)
        next_speech_token = torch.multinomial(probs, num_samples=1)

        generated_speech_tokens.append(next_speech_token)
        current_speech_token = next_speech_token

        for _ in tqdm(range(max_gen_len)):
            current_speech_embed = self.speech_emb(current_speech_token)

            llm_outputs = self.tfmr(
                inputs_embeds=current_speech_embed,
                past_key_values=past_key_values,
                use_cache=True
            )

            hidden_states = llm_outputs[0]
            past_key_values = llm_outputs.past_key_values
            speech_logits = self.speech_head(hidden_states)

            input_ids = torch.cat(generated_speech_tokens, dim=1)
            processed_logits = logits_processors(input_ids, speech_logits[:, -1, :])
            if torch.all(processed_logits == -float("inf")):
                print("Warning: All logits are -inf")
                break

            probs = F.softmax(processed_logits, dim=-1)
            next_speech_token = torch.multinomial(probs, num_samples=1)

            generated_speech_tokens.append(next_speech_token)
            current_speech_token = next_speech_token
            if torch.all(next_speech_token == self.hp.stop_speech_token):
                break

        all_tokens = torch.cat(generated_speech_tokens, dim=1)

        # Remove EOS token if present
        if all_tokens.size(1) > 0 and all_tokens[0, -1] == self.hp.stop_speech_token:
            all_tokens = all_tokens[:, :-1]

        return all_tokens

    @torch.inference_mode()
    def inference_fast(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: Tensor,
        max_new_tokens: int = 1000,
        max_cache_len: int = TOKEN_LIMIT,
        temperature: float = 0.8,
        cfg_weight: float = 0.5,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        top_p: float = 1.0,
    ):
        """Fast inference using CUDA graphs and StaticCache. No AlignmentStreamAnalyzer.

        Falls back to regular inference() on non-CUDA devices.
        Note: cfg_weight, temperature, and max_cache_len must remain constant across
        calls for CUDA graph reuse to be valid.
        """
        if self.device.type != "cuda":
            logger.info("generate_fast() called on non-CUDA device (%s) — falling back to generate()", self.device)
            return self.inference(
                t3_cond=t3_cond,
                text_tokens=text_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )

        # Build the fast-path backend once (separate from inference()'s self.patched_model)
        if not hasattr(self, '_patched_model_fast'):
            self._patched_model_fast = T3HuggingfaceBackend(
                config=self.cfg,
                llama=self.tfmr,
                speech_enc=self.speech_emb,
                speech_head=self.speech_head,
                alignment_stream_analyzer=None,  # disabled for fast path
            )
            # Re-init processors now that self.device is guaranteed CUDA.
            # init_processors() is also called in __init__ (on CPU), so this moves
            # the warper tensors to the correct device for graph capture.
            self.init_processors()

        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)

        initial_speech_tokens = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _ = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech_tokens,
            cfg_weight=cfg_weight,
        )

        # Pre-compute embedding caches (needed for bos_embed and the generation loop)
        self.get_speech_pos_embedding_cache(max_new_tokens + 1, dtype=embeds.dtype)
        self.init_speech_embedding_cache(self.hp.speech_tokens_dict_size, dtype=embeds.dtype)

        bos_token = torch.tensor([[self.hp.start_speech_token]], dtype=torch.long, device=self.device)
        bos_embed = self._speech_embedding_cache[bos_token] + self._speech_pos_embedding_cache[0]
        if cfg_weight > 0.0:
            bos_embed = torch.cat([bos_embed, bos_embed])
        inputs_embeds = torch.cat([embeds, bos_embed], dim=1)
        seq_len = inputs_embeds.shape[1]  # actual context length, before padding

        # Pad to TOKEN_LIMIT for static shape (required for CUDA graph capture)
        pad_len = TOKEN_LIMIT - inputs_embeds.shape[1]
        if pad_len > 0:
            pad = torch.zeros(
                inputs_embeds.shape[0], pad_len, inputs_embeds.shape[2],
                dtype=inputs_embeds.dtype, device=self.device,
            )
            inputs_embeds = torch.cat([inputs_embeds, pad], dim=1)

        effective_batch = 2 if cfg_weight > 0.0 else 1
        kv_cache = self.get_or_reset_static_cache(
            max_batch_size=effective_batch, max_cache_len=max_cache_len
        )

        self.update_processors(top_p=top_p, min_p=min_p, repetition_penalty=float(repetition_penalty))

        # cfg_weight and temperature are baked into captured CUDA graphs — changing them
        # between calls would produce silently wrong output. Detect the change and reset.
        if hasattr(self, '_cudagraph_wrapper'):
            if cfg_weight != getattr(self, '_last_cfg_weight', None) or \
                    temperature != getattr(self, '_last_temperature', None):
                self._cudagraph_wrapper.reset()
                del self._cudagraph_wrapper
        self._last_cfg_weight = cfg_weight
        self._last_temperature = temperature

        # Sync processor refs on existing wrapper (update_processors may have replaced them)
        if hasattr(self, '_cudagraph_wrapper'):
            self._cudagraph_wrapper.repetition_penalty_processor = self.repetition_penalty_processor
            self._cudagraph_wrapper.min_p_warper = self.min_p_warper
            self._cudagraph_wrapper.top_p_warper = self.top_p_warper

        # Initial forward pass: fills KV cache with the full prompt context
        output_logits = _fast_initial_forward_pass(
            inputs_embeds, kv_cache, self._patched_model_fast, seq_len
        )

        # Create CUDA graph wrapper once; reuse on subsequent calls
        if not hasattr(self, '_cudagraph_wrapper'):
            self._cudagraph_wrapper = T3StepCUDAGraphWrapper(
                _fast_generate_t3_token,
                self._patched_model_fast,
                kv_cache,
                self.repetition_penalty_processor,
                self.min_p_warper,
                self.top_p_warper,
            )

        # generated_ids: batch_size=1 (we only track the conditional sequence)
        bos_len = bos_token.shape[1]
        generated_ids = torch.full(
            (1, bos_len + TOKEN_LIMIT), 0, dtype=torch.long, device=self.device
        )
        generated_ids[0, :bos_len] = bos_token

        batch_idx = torch.zeros(1, dtype=torch.long, device=self.device)
        indices = torch.arange(1, max_new_tokens + 1, device=self.device)

        stop_token = torch.tensor(self.hp.stop_speech_token, device=self.device)
        length_hint = text_tokens.shape[1] * 2

        for i in tqdm(range(max_new_tokens), desc="Sampling (fast)", dynamic_ncols=True):
            i_tensor = indices[i]
            torch.compiler.cudagraph_mark_step_begin()
            max_position = get_next_bucket(i + seq_len)
            outputs = self._cudagraph_wrapper(
                self._speech_embedding_cache,
                output_logits,
                i_tensor,
                batch_idx,
                self._speech_pos_embedding_cache,
                generated_ids,
                cfg_weight,
                temperature,
                stride_length=1,
                max_position=max_position,
            )
            output_logits = outputs[1].clone()
            generated_ids = outputs[2].clone()

            if i > length_hint and i % 20 == 0:
                if (generated_ids[0] == stop_token).any():
                    break

        return generated_ids


def _fast_initial_forward_pass(inputs_embeds, kv_cache, patched_model, seq_len):
    """Run the full prompt through the model to fill the KV cache."""
    inputs_embeds = inputs_embeds[:, :seq_len, :]
    cache_position = torch.arange(seq_len, device=inputs_embeds.device)
    out = patched_model(
        inputs_embeds=inputs_embeds,
        past_key_values=kv_cache,
        cache_position=cache_position,
        output_hidden_states=False,
        output_attentions=False,
    )
    return out.logits[:, -1:, :]


def _fast_generate_t3_token(
    speech_embedding_cache,
    output_logits,
    i_tensor,
    batch_idx,
    speech_pos_embedding_cache,
    generated_ids,
    cfg_weight,
    temperature,
    repetition_penalty_processor,
    min_p_warper,
    top_p_warper,
    patched_model,
    kv_cache,
    stride_length=1,
    max_position=None,
    alignment_stream_analyzer=None,
):
    """Single-token generation step; designed to be wrapped in a CUDA graph."""
    logits = output_logits[:, -1, :]
    if cfg_weight > 0.0:
        logits = logits[0:1] + cfg_weight * (logits[0:1] - logits[1:2])

    logits = repetition_penalty_processor(generated_ids, logits)
    if temperature != 1.0:
        logits = logits / temperature
    logits = min_p_warper(None, logits)
    logits = top_p_warper(None, logits)

    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    generated_ids.index_put_((batch_idx, i_tensor), next_token.squeeze(-1))

    position_embed = torch.index_select(speech_pos_embedding_cache, 0, i_tensor).squeeze(0)
    next_token_embed = speech_embedding_cache[next_token] + position_embed
    if cfg_weight > 0.0:
        next_token_embed = torch.cat([next_token_embed, next_token_embed])

    out = patched_model(
        inputs_embeds=next_token_embed,
        past_key_values=kv_cache,
        cache_position=kv_cache.get_seq_length().unsqueeze(0),
        max_position=max_position,
        output_hidden_states=False,
        output_attentions=False,
    )
    return next_token, out.logits
