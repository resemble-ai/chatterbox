from typing import Optional

import torch
from torch import nn as nn
from .custom_llama.modeling_llama import LlamaConfig, LlamaModel, LlamaPreTrainedModel, GenerationMixin
from transformers.cache_utils import StaticCache
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
        past_key_values: StaticCache,
        use_cache=True,
        output_attentions=False,
        cache_position=None,
    ):
        """
        This is a method used by huggingface's generate() method.
        Overridden here to apply our custom layer norm and speech logit projection layers.

        :param inputs_embeds: (B, S, C) float32 tensor of conditioning inputs. If past key values are given,
        S should be 1.
        """
        # Handle input validation before calling the model

        tfmr_out = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=False,
            cache_position=cache_position,
        )
        # Top-level sompilation may require .clone() here
        hidden_states = tfmr_out[0]

        logits = self.speech_head(hidden_states)
        # assert inputs_embeds.size(0) == 1 # (disabled for CFG)

        # NOTE: hallucination handler may modify logits to force emit an EOS token
        # logits = self.alignment_stream_analyzer.step(logits)

        return logits
