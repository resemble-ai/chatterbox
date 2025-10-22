LLAMA_520M_CONFIG_DICT = dict(
    # Arbitrary small number that won't cause problems when loading.
    # These param are unused due to custom input layers.
    vocab_size=8,
    # default params needed for loading most pretrained 1B weights
    max_position_embeddings=131072,
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=30,
    num_attention_heads=16,
    attn_implementation="eager",
    head_dim=64,
    tie_word_embeddings=False,
    hidden_act="silu",
    attention_bias=False,
    attention_dropout=0.0,
    initializer_range=0.02,
    mlp_bias=False,
    model_type="llama",
    num_key_value_heads=16,
    pretraining_tp=1,
    rms_norm_eps=1e-05,
    rope_scaling=dict(
        factor=8.0,
        high_freq_factor=4.0,
        low_freq_factor=1.0,
        original_max_position_embeddings=8192,
        rope_type="llama3"
    ),
    rope_theta=500000.0,
    torch_dtype="bfloat16",
    use_cache=True,
)

LLAMA_CONFIGS = {
    "Llama_520M": LLAMA_520M_CONFIG_DICT,
}
