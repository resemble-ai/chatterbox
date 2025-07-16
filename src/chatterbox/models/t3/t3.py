# Copyright (c) 2025 Resemble AI
# MIT License
import logging
from typing import Union, Optional, List

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn, Tensor
# from transformers import LlamaModel, LlamaConfig, DynamicCache
from .inference.custom_llama.modeling_llama import LlamaModel, LlamaConfig
from transformers.cache_utils import StaticCache
from transformers.generation.logits_process import MinPLogitsWarper, TopPLogitsWarper, RepetitionPenaltyLogitsProcessor

from .modules.learned_pos_emb import LearnedPositionEmbeddings

from .modules.cond_enc import T3CondEnc, T3Cond
from .modules.t3_config import T3Config
from .llama_configs import LLAMA_CONFIGS
from .inference.t3_hf_backend import T3HuggingfaceBackend
from .inference.alignment_stream_analyzer import AlignmentStreamAnalyzer


logger = logging.getLogger(__name__)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


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

    def __init__(self, hp=T3Config()):
        super().__init__()
        self.hp = hp
        self.cfg = LlamaConfig(**LLAMA_CONFIGS[hp.llama_config_name])
        self.tfmr = LlamaModel(self.cfg)
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

    def init_patched_model(self):
        # TODO? synchronize the expensive compile function
        # with self.compile_lock:
        if not self.compiled:
            # alignment_stream_analyzer = AlignmentStreamAnalyzer(
            #     self.tfmr,
            #     None,
            #     text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
            #     alignment_layer_idx=9, # TODO: hparam or something?
            #     eos_idx=self.hp.stop_speech_token,
            # )
            patched_model = T3HuggingfaceBackend(
                config=self.cfg,
                llama=self.tfmr,
                speech_enc=self.speech_emb,
                speech_head=self.speech_head,
                # alignment_stream_analyzer=alignment_stream_analyzer,
            )
            self.patched_model = patched_model
            self.compiled = True

    def get_cache(self, config, max_batch_size, max_cache_len, device, dtype):
        if hasattr(self, 'backend_cache'):
            if self.backend_cache_params['max_batch_size'] == max_batch_size and \
                self.backend_cache_params['max_cache_len'] == max_cache_len and \
                self.backend_cache_params['dtype'] == dtype and \
                self.backend_cache_params['device'] == device:
                self.backend_cache.reset()
                return self.backend_cache
            else:
                del self.backend_cache

        cache = StaticCache(
            config=config,
            max_batch_size=max_batch_size,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
        )
        # save parameters in t3 since huggingface fails deprecation standards.
        self.backend_cache_params = {
            'max_batch_size': max_batch_size,
            'max_cache_len': max_cache_len,
            'device': device,
            'dtype': dtype,
        }
        self.backend_cache = cache
        return cache

    def get_speech_pos_embedding_cache(self, max_gen_tokens, dtype):
        if not hasattr(self, '_speech_pos_embedding_cache') or self._speech_pos_embedding_cache.size(0) < max_gen_tokens:
            # Create cache with embeddings for positions 0 to max_gen_tokens-1
            self._speech_pos_embedding_cache = []
            for pos in range(max_gen_tokens):
                embedding = self.speech_pos_emb.get_fixed_embedding(pos)
                self._speech_pos_embedding_cache.append(embedding)
            # Stack and move to device
            self._speech_pos_embedding_cache = torch.stack(self._speech_pos_embedding_cache, dim=0).to(device=self.device)
        elif self._speech_pos_embedding_cache.dtype != dtype:
            self._speech_pos_embedding_cache = self._speech_pos_embedding_cache.to(dtype=dtype)
        return self._speech_pos_embedding_cache

    def init_speech_embedding_cache(self, vocab_size, dtype):
        if not hasattr(self, '_speech_embedding_cache') or self._speech_embedding_cache.size(0) < vocab_size:
            # Create cache with embeddings for positions 0 to max_gen_tokens-1
            self._speech_embedding_cache = []
            for pos in range(vocab_size):
                pos = torch.tensor([pos], device=self.device)
                embedding = self.speech_emb(pos)
                self._speech_embedding_cache.append(embedding.squeeze(0))
            # Stack and move to device
            self._speech_embedding_cache = torch.stack(self._speech_embedding_cache, dim=0).to(device=self.device)
        elif self._speech_embedding_cache.dtype != dtype:
            self._speech_embedding_cache = self._speech_embedding_cache.to(dtype=dtype)
        return self._speech_embedding_cache

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
        min_p=0.05,
        top_p=1.0,
        length_penalty=1.0,
        repetition_penalty=1.2,
        cfg_weight=0,
        max_cache_len=None,
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
        self.init_patched_model()
        # Pre-compute embeddings cache for the generation loop
        self.get_speech_pos_embedding_cache(max_new_tokens + 1 or self.hp.max_speech_tokens, dtype=embeds.dtype)
        self.init_speech_embedding_cache(vocab_size=self.hp.speech_tokens_dict_size, dtype=embeds.dtype)

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
        #     top_p=top_p,
        #     length_penalty=length_penalty,
        #     repetition_penalty=repetition_penalty,
        #     do_sample=do_sample,
        #     # cache_implementation=None if not self.compiled else "static",
        # )

        device = embeds.device

        bos_token = torch.tensor([[self.hp.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = self._speech_embedding_cache[bos_token]
        bos_embed = bos_embed + self._speech_pos_embedding_cache[0]

        # batch_size=2 for CFG
        bos_embed = torch.cat([bos_embed, bos_embed])

        # Combine condition and BOS token for the initial input if cfg_weight > 0
        if cfg_weight > 0:
            inputs_embeds = torch.cat([embeds, bos_embed], dim=1)
        else:
            inputs_embeds = embeds

        # Track generated token ids; start with the BOS token.
        PAD_TOKEN_ID = self.hp.stop_speech_token + 1 # Assuming unused
        bos_len = bos_token.shape[1]
        # using batch size of 1, otherwise use generated_ids[:, i] 
        generated_ids = torch.full((1, bos_len + max_new_tokens), PAD_TOKEN_ID, dtype=torch.long, device=device)
        generated_ids[0, :bos_len] = bos_token

        predicted = []  # To store the predicted tokens

        # Instantiate the logits processors.
        top_p_warper = TopPLogitsWarper(top_p=top_p)
        min_p_warper = MinPLogitsWarper(min_p=min_p)
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)

        # move all inputs to patched_model.dtype
        inputs_embeds = inputs_embeds.to(self.patched_model.dtype)
        embeds = embeds.to(self.patched_model.dtype)
        bos_embed = bos_embed.to(self.patched_model.dtype)

        stop_token_tensor = torch.tensor(self.hp.stop_speech_token, device=self.device)
        
        # Fix: Set max_batch_size based on CFG usage
        effective_batch_size = 2 if cfg_weight > 0.0 else 1

        _, seq_len = inputs_embeds.shape[:2]
        assert max_cache_len > seq_len + max_new_tokens, \
            f"max_cache_len {max_cache_len} is too small for seq_len {seq_len} and max_new_tokens {max_new_tokens}"

        kv_cache = self.get_cache(
            config=self.patched_model.config,
            max_batch_size=effective_batch_size,
            max_cache_len=max_cache_len,
            device=self.patched_model.device,
            dtype=self.patched_model.dtype,
        )

        # Move check higher to avoid polluting the loop
        assert not kv_cache.get_seq_length() > 0, \
            "Cannot process large input when cache already has content"

        length_guesstimate = text_tokens.shape[1] * 2
        print(f"Estimated token count: {length_guesstimate}")
        length_guesstimate = int(length_guesstimate * 0.8) # variance

        cache_position = torch.arange(seq_len, device=inputs_embeds.device)

        # ---- Initial Forward Pass (no kv_cache yet) ----
        output_logits = self.patched_model(
            inputs_embeds=inputs_embeds,
            past_key_values=kv_cache,
            cache_position=cache_position,
        )

        # ---- Generation Loop using kv_cache ----
        for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True):
            logits = output_logits[:, -1, :]

            # CFG
            if cfg_weight > 0.0:
                logits_cond = logits[0:1]
                logits_uncond = logits[1:2]
                logits = logits_cond + cfg_weight * (logits_cond - logits_uncond)

            logits = logits.squeeze(1)

            # Apply temperature scaling.
            if temperature != 1.0:
                logits = logits / temperature

            # Apply repetition penalty and top‑p filtering.
            logits = repetition_penalty_processor(generated_ids, logits)
            logits = min_p_warper(None, logits)
            logits = top_p_warper(None, logits)

            # Convert logits to probabilities and sample the next token.
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # shape: (B, 1)

            predicted.append(next_token)
            generated_ids[0, i + bos_len] = next_token

            # if i % tokens_per_slice == 0:
            #     yield torch.cat(predicted, dim=1)

            # Get embedding for the new token.
            next_token_embed = self._speech_embedding_cache[next_token] + self._speech_pos_embedding_cache[i + 1]

            #  For CFG
            if cfg_weight > 0.0:
                next_token_embed = torch.cat([next_token_embed, next_token_embed])

            # Check for EOS token.
            if i > length_guesstimate and i % 20 == 0:
                if (generated_ids == stop_token_tensor).any():
                    break

            # Forward pass with only the new token and the cached past.
            torch.compiler.cudagraph_mark_step_begin()
            output_logits = self._step_compilation_target(
                next_token_embed,
                kv_cache,
            )

        
        return torch.cat(predicted, dim=1)

    #@torch.compile(backend="cudagraphs", fullgraph=True)
    def _step_compilation_target(
        self,
        next_token_embed: Tensor,
        kv_cache: StaticCache,
    ):
        return self.patched_model(
            inputs_embeds=next_token_embed,
            past_key_values=kv_cache,
            cache_position=kv_cache.get_seq_length().unsqueeze(0),
        )
