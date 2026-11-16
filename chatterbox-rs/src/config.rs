#[derive(Debug, Clone, PartialEq)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub attn_implementation: String,
    pub head_dim: usize,
    pub tie_word_embeddings: bool,
    pub hidden_act: String,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub initializer_range: f64,
    pub mlp_bias: bool,
    pub model_type: String,
    pub num_key_value_heads: usize,
    pub pretraining_tp: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub use_cache: bool,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            vocab_size: 8,
            max_position_embeddings: 131072,
            hidden_size: 1024,
            intermediate_size: 4096,
            num_hidden_layers: 30,
            num_attention_heads: 16,
            attn_implementation: "sdpa".to_string(),
            head_dim: 64,
            tie_word_embeddings: false,
            hidden_act: "silu".to_string(),
            attention_bias: false,
            attention_dropout: 0.0,
            initializer_range: 0.02,
            mlp_bias: false,
            model_type: "llama".to_string(),
            num_key_value_heads: 16,
            pretraining_tp: 1,
            rms_norm_eps: 1e-05,
            rope_theta: 500000.0,
            use_cache: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct T3Config {
    pub start_text_token: u32,
    pub stop_text_token: u32,
    pub text_tokens_dict_size: usize,
    pub max_text_tokens: usize,

    pub start_speech_token: u32,
    pub stop_speech_token: u32,
    pub speech_tokens_dict_size: usize,
    pub max_speech_tokens: usize,

    pub llama_config_name: String,
    pub input_pos_emb: String,
    pub speech_cond_prompt_len: usize,

    pub encoder_type: String,
    pub speaker_embed_size: usize,
    pub use_perceiver_resampler: bool,
    pub emotion_adv: bool,

    pub llama_config: LlamaConfig,
}

impl T3Config {
    pub fn english_only() -> Self {
        Self {
            start_text_token: 255,
            stop_text_token: 0,
            text_tokens_dict_size: 704,
            max_text_tokens: 2048,
            start_speech_token: 6561,
            stop_speech_token: 6562,
            speech_tokens_dict_size: 8194,
            max_speech_tokens: 4096,
            llama_config_name: "Llama_520M".to_string(),
            input_pos_emb: "learned".to_string(),
            speech_cond_prompt_len: 150,
            encoder_type: "voice_encoder".to_string(),
            speaker_embed_size: 256,
            use_perceiver_resampler: true,
            emotion_adv: true,
            llama_config: LlamaConfig::default(),
        }
    }

    pub fn multilingual() -> Self {
        let mut config = Self::english_only();
        config.text_tokens_dict_size = 2454;
        config
    }

    pub fn n_channels(&self) -> usize {
        self.llama_config.hidden_size
    }

    pub fn is_multilingual(&self) -> bool {
        self.text_tokens_dict_size == 2454
    }
}
