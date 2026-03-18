import logging
from typing import Optional

import torch
from transformers import PretrainedConfig

from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.t3.t3 import T3, T3Cond

logger = logging.getLogger(__name__)

# Fields from T3Config to expose via the HF-compatible PretrainedConfig.
_T3_CONFIG_FIELDS = (
    "llama_config_name",
    "text_tokens_dict_size",
    "speech_tokens_dict_size",
    "max_text_tokens",
    "max_speech_tokens",
    "speech_cond_prompt_len",
    "start_text_token",
    "stop_text_token",
    "start_speech_token",
    "stop_speech_token",
)


class _HFCompatibleConfig(PretrainedConfig):
    """Minimal PretrainedConfig wrapper so the HF Trainer can serialise model config."""

    model_type = "chatterbox_t3_finetune"


class T3ForFineTuning(torch.nn.Module):
    """Wraps the upstream T3 model to match the HuggingFace Trainer interface.

    The HF Trainer expects:
      - ``model.config`` to be a ``PretrainedConfig`` instance.
      - ``forward()`` to return ``(loss, ...)``.
    """

    def __init__(self, t3_model: T3, chatterbox_t3_config: T3Config):
        super().__init__()
        self.t3 = t3_model
        self.chatterbox_t3_config = chatterbox_t3_config

        config = _HFCompatibleConfig()
        for field in _T3_CONFIG_FIELDS:
            setattr(config, field, getattr(chatterbox_t3_config, field))
        self.config = config

    def forward(
        self,
        text_tokens,
        text_token_lens,
        speech_tokens,
        speech_token_lens,
        t3_cond_speaker_emb,
        t3_cond_prompt_speech_tokens,
        t3_cond_emotion_adv,
        labels_text: Optional[torch.LongTensor] = None,
        labels_speech: Optional[torch.LongTensor] = None,
        **kwargs,  # Absorbs extra collator fields (e.g. language_ids) not used by the model.
    ):
        """Compute next-token prediction loss for text and speech streams."""
        t3_cond = T3Cond(
            speaker_emb=t3_cond_speaker_emb,
            cond_prompt_speech_tokens=t3_cond_prompt_speech_tokens,
            cond_prompt_speech_emb=None,
            emotion_adv=t3_cond_emotion_adv,
        )

        assert labels_text is not None and labels_speech is not None, ("labels_text and labels_speech must be provided during training")

        loss_text, loss_speech, speech_logits = self.t3.loss_new(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            labels_text=labels_text,
            labels_speech=labels_speech,
        )

        total_loss = loss_text + loss_speech
        return total_loss, speech_logits
