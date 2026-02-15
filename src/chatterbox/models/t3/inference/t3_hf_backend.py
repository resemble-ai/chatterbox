from typing import Optional

import torch
from torch import nn as nn
from transformers import LlamaConfig, LlamaModel, LlamaPreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class T3HuggingfaceBackend(LlamaPreTrainedModel, GenerationMixin):
    """
    Override some HuggingFace interface methods so we can use the standard `generate` method with our
    custom embedding / logit layers.

    NOTE: need to extend "*PreTrainedModel" to avoid re-initializing weights!
    """

    def __init__(
        self,
        config: LlamaConfig,
        llama: LlamaModel,
        *,
        speech_enc,
        speech_head,
        latents_queue=None,
        logits_queue=None,
        alignment_stream_analyzer: 'AlignmentStreamAnalyzer'=None,
    ):
        super().__init__(config)
        self.model = llama
        self.speech_enc = speech_enc
        self.speech_head = speech_head
        self._added_cond = False
        self.alignment_stream_analyzer = alignment_stream_analyzer

    @torch.inference_mode()
    def prepare_inputs_for_generation(
        self, input_ids: torch.Tensor, decoder_cond: torch.Tensor, use_cache: bool, past_key_values=None,
        # This argument was introduced in some recent version of transformers (>=4.29.1)
        cache_position=None
    ):
        """
        This is a method used by huggingface's generate() method.
        Overridden here to apply our custom speech token embedding layer.

        :param input_ids: (B, S) int64 tensors of input tokens.
        :param decoder_cond: (B, T, C) float32 tensor of conditioning (prefixed to <input_embeds>)
        """

        # Make use of the kv cache: only the last input ID is new, we trim away all the ones before
        if not use_cache:
            past_key_values = None
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # custom speech token embedding layer
        inputs_embeds = self.speech_enc(input_ids)

        # prefix decoder conditioning if applicable
        if not self._added_cond:
            assert past_key_values is not None # should be first step
            if decoder_cond.size(0) != inputs_embeds.size(0):
                decoder_cond = decoder_cond.expand(inputs_embeds.size(0), -1, -1)
            inputs_embeds = torch.cat([decoder_cond, inputs_embeds], dim=1)
            self._added_cond = True

        return {
            "inputs_embeds": inputs_embeds,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @torch.inference_mode()
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,   # ✅ default False
        return_dict: bool = True,
    ):
        """
        Overridden HF forward to apply speech_head on transformer hidden state.
    
        IMPORTANT:
        - We do NOT require output_hidden_states=True.
        - We compute logits from last_hidden_state (always available) to avoid
          materializing hidden_states lists every step (big speed win).
        """
        is_large_input = inputs_embeds.size(1) != 1
        has_cache = past_key_values is not None and len(past_key_values) > 0
        assert not (is_large_input and has_cache), "Cannot pass long inputs when cache is already populated"
        assert return_dict, "This backend expects return_dict=True"
    
        tfmr_out = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
    
        # ✅ Always available on HF models (LlamaModel returns this)
        last_h = tfmr_out.last_hidden_state  # (B, seq, dim)
        logits = self.speech_head(last_h)
    
        return CausalLMOutputWithCrossAttentions(
            logits=logits,
            past_key_values=tfmr_out.past_key_values,
            hidden_states=(tfmr_out.hidden_states if output_hidden_states else None),
            attentions=(tfmr_out.attentions if output_attentions else None),
        )
