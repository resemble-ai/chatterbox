from ..llama_configs import LLAMA_CONFIGS


class T3Config:
    def __init__(self, text_tokens_dict_size=704, kv_cache_dtype="float16"):
        self.start_text_token = 255
        self.stop_text_token = 0
        self.text_tokens_dict_size = text_tokens_dict_size
        self.max_text_tokens = 2048

        self.start_speech_token = 6561
        self.stop_speech_token = 6562
        self.speech_tokens_dict_size = 8194
        self.max_speech_tokens = 4096

        self.llama_config_name = "Llama_520M"
        self.input_pos_emb = "learned"
        self.speech_cond_prompt_len = 150

        self.encoder_type = "voice_encoder"
        self.speaker_embed_size = 256
        self.use_perceiver_resampler = True
        self.emotion_adv = True

        # KV cache dtype optimization: 'float16' (default, recommended) or None for full precision
        # Float16 KV cache provides 18-32% speed improvement with significant memory savings
        self.kv_cache_dtype = kv_cache_dtype

    @property
    def n_channels(self):
        return LLAMA_CONFIGS[self.llama_config_name]["hidden_size"]

    @property
    def is_multilingual(self):
        return self.text_tokens_dict_size == 2454

    @classmethod
    def english_only(cls, kv_cache_dtype="float16"):
        """Create configuration for English-only TTS model.

        Args:
            kv_cache_dtype: KV cache dtype - 'float16' (default, recommended) or None for full precision.
                          Float16 provides 18-32% speed improvement with significant memory savings.
        """
        return cls(text_tokens_dict_size=704, kv_cache_dtype=kv_cache_dtype)

    @classmethod
    def multilingual(cls, kv_cache_dtype="float16"):
        """Create configuration for multilingual TTS model.

        Args:
            kv_cache_dtype: KV cache dtype - 'float16' (default, recommended) or None for full precision.
                          Float16 provides 18-32% speed improvement with significant memory savings.
        """
        return cls(text_tokens_dict_size=2454, kv_cache_dtype=kv_cache_dtype)
