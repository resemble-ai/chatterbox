"""PoC: load T3MLX weights, run forward pass, measure tok/s."""
import time
import numpy as np
import mlx.core as mx
from src.chatterbox.models.t3_mlx import T3MLX, T3MLXConfig


def load_weights(model, path="assets/t3_mlx_weights.npz"):
    w = mx.load(path)
    model.load_weights(list(w.items()))
    n_loaded = len(w)
    print(f"Loaded {n_loaded} weight tensors")


def benchmark_forward(model, seq_len=100, n_warmup=3, n_iter=20):
    B = 1
    inputs_embeds = mx.random.normal((B, seq_len, model.config.dim), dtype=mx.float32)
    print(f"Input shape: {inputs_embeds.shape}")

    for _ in range(n_warmup):
        logits, kvs = model.forward(inputs_embeds)

    start = time.perf_counter()
    for _ in range(n_iter):
        logits, kvs = model.forward(inputs_embeds)
    end = time.perf_counter()

    avg_ms = (end - start) / n_iter * 1000
    print(f"Forward ({seq_len} tokens): {avg_ms:.1f} ms | {1000/avg_ms:.0f} tok/s")

    B = 1
    inputs_embeds = mx.random.normal((B, 1, model.config.dim), dtype=mx.float32)
    _, kvs = model.forward(mx.random.normal((B, seq_len, model.config.dim)))
    for _ in range(n_warmup):
        logits, kvs = model.forward(inputs_embeds, past_kvs=kvs)

    start = time.perf_counter()
    for _ in range(n_iter):
        logits, kvs = model.forward(inputs_embeds, past_kvs=kvs)
    end = time.perf_counter()

    avg_ms_step = (end - start) / n_iter * 1000
    print(f"Autoregressive step (1 token): {avg_ms_step:.1f} ms | {1000/avg_ms_step:.0f} tok/s")


def benchmark_ar_loop(model, cond_len=34, text_len=50, n_tokens=200):
    B = 1
    cond_emb = mx.random.normal((B, cond_len, model.config.dim))
    text_tokens = mx.random.randint(0, model.config.text_vocab_size, (B, text_len))

    start = time.perf_counter()
    tokens = model.generate(cond_emb, text_tokens, max_new_tokens=n_tokens, seed=0)
    elapsed = time.perf_counter() - start

    n_gen = tokens.shape[0]
    print(f"Generated {n_gen} tokens in {elapsed:.2f}s = {n_gen/elapsed:.1f} tok/s")
    return tokens
