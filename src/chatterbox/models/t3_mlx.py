import math
import mlx.core as mx
import mlx.nn as nn

_BACKBONE_KEYS = {f"layers.{i}.{sub}" for i in range(30) for sub in [
    "attention_norm.weight",
    "ffn_norm.weight",
    "attention.q_proj.weight", "attention.k_proj.weight",
    "attention.v_proj.weight", "attention.o_proj.weight",
    "mlp.gate.weight", "mlp.up.weight", "mlp.down.weight",
]} | {"norm.weight",     "speech_emb.weight", "speech_head.weight",
    "text_emb.weight", "text_head.weight",
    "speech_pos_emb.weight", "text_pos_emb.weight"}


def llama3_rope_freqs(dim, base=500000.0, original_max_pos=8192,
                      scaling_factor=8.0, high_freq_factor=4.0, low_freq_factor=1.0):
    indices = mx.arange(0, dim, 2, dtype=mx.float32)
    inv_freq = 1.0 / (base ** (indices / dim))

    low_wavelen = original_max_pos / low_freq_factor
    high_wavelen = original_max_pos / high_freq_factor
    wavelen = 2.0 * math.pi / inv_freq
    smooth = (wavelen - high_wavelen) / (low_wavelen - high_wavelen)
    smooth = mx.clip(smooth, 0.0, 1.0)

    scaled = (1.0 - smooth) * (inv_freq / scaling_factor) + smooth * inv_freq
    return scaled


_ROPE_FREQS = llama3_rope_freqs(64)


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, bias: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.o_proj = nn.Linear(dim, dim, bias=bias)

    def __call__(self, x, mask=None, past_kv=None):
        B, L, D = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        if past_kv is not None:
            pk, pv = past_kv
            cache_len = pk.shape[2]
            q = mx.fast.rope(q, dims=self.head_dim, traditional=False,
                             base=None, scale=1.0, offset=cache_len, freqs=_ROPE_FREQS)
            k = mx.fast.rope(k, dims=self.head_dim, traditional=False,
                             base=None, scale=1.0, offset=cache_len, freqs=_ROPE_FREQS)
            k = mx.concatenate([pk, k], axis=2)
            v = mx.concatenate([pv, v], axis=2)
        else:
            q = mx.fast.rope(q, dims=self.head_dim, traditional=False,
                             base=None, scale=1.0, offset=0, freqs=_ROPE_FREQS)
            k = mx.fast.rope(k, dims=self.head_dim, traditional=False,
                             base=None, scale=1.0, offset=0, freqs=_ROPE_FREQS)

        scale = 1.0 / math.sqrt(self.head_dim)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        out = self.o_proj(out)
        return out, (k, v)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.up = nn.Linear(dim, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, dim, bias=bias)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class LLaMADecoderLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.attention_norm = nn.RMSNorm(dim)
        self.attention = Attention(dim, n_heads)
        self.ffn_norm = nn.RMSNorm(dim)
        self.mlp = SwiGLU(dim, dim * mlp_ratio)

    def __call__(self, x, mask=None, past_kv=None):
        r = x
        x = self.attention_norm(x)
        x, kv = self.attention(x, mask=mask, past_kv=past_kv)
        x = r + x

        r = x
        x = self.ffn_norm(x)
        x = self.mlp(x)
        x = r + x
        return x, kv


class T3MLXConfig:
    dim = 1024
    n_layers = 30
    n_heads = 16
    mlp_ratio = 4
    vocab_size = 8194
    text_vocab_size = 704
    max_text_tokens = 2048
    max_speech_tokens = 4096
    text_pos_len = max_text_tokens + 2
    speech_pos_len = max_speech_tokens + 4
    start_speech_token = 6561
    stop_speech_token = 6562


class T3MLX(nn.Module):
    def __init__(self, config: T3MLXConfig = None):
        super().__init__()
        self.config = config or T3MLXConfig()
        cfg = self.config

        self.text_emb = nn.Embedding(cfg.text_vocab_size, cfg.dim)
        self.speech_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.text_pos_emb = nn.Embedding(cfg.text_pos_len, cfg.dim)
        self.speech_pos_emb = nn.Embedding(cfg.speech_pos_len, cfg.dim)
        self.text_head = nn.Linear(cfg.dim, cfg.text_vocab_size, bias=False)
        self.speech_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        self.layers = [LLaMADecoderLayer(cfg.dim, cfg.n_heads, cfg.mlp_ratio) for _ in range(cfg.n_layers)]
        self.norm = nn.RMSNorm(cfg.dim)

    def embed_text(self, text_tokens):
        emb = self.text_emb(text_tokens)
        n = text_tokens.shape[1]
        pos = self.text_pos_emb(mx.arange(n, dtype=mx.int64))
        return emb + pos

    def embed_speech(self, speech_tokens):
        return self.speech_emb(speech_tokens)

    def embed_speech_with_pos(self, speech_tokens, offset=0):
        emb = self.speech_emb(speech_tokens)
        n = speech_tokens.shape[1]
        pos = self.speech_pos_emb(mx.arange(offset, offset + n, dtype=mx.int64))
        return emb + pos

    def forward(self, inputs_embeds, past_kvs=None, mask=None):
        h = inputs_embeds
        new_kvs = []
        for i, layer in enumerate(self.layers):
            pkv = past_kvs[i] if past_kvs is not None else None
            h, kv = layer(h, mask=mask, past_kv=pkv)
            new_kvs.append(kv)
        h = self.norm(h)
        logits = self.speech_head(h)
        return logits, new_kvs

    def generate(
        self,
        cond_emb,
        text_tokens,
        max_new_tokens=500,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.2,
        seed=42,
    ):
        mx.random.seed(seed)

        text_emb = self.embed_text(text_tokens)

        bos_token = mx.array([[self.config.start_speech_token]], dtype=mx.int32)
        speech_emb = self.embed_speech_with_pos(bos_token)

        inputs_embeds = mx.concatenate([cond_emb, text_emb, speech_emb], axis=1)

        L = inputs_embeds.shape[1]
        causal_mask = mx.tril(mx.ones((L, L), dtype=mx.bool_))
        logits, past_kvs = self.forward(inputs_embeds, past_kvs=None, mask=causal_mask)
        logits = logits[:, -1, :].squeeze(0)

        generated_ids = []

        for step in range(max_new_tokens):
            logits = self._sample_logits(logits, temperature, repetition_penalty, generated_ids)
            probs = mx.softmax(logits, axis=-1)
            next_token = self._top_p_sample(probs, top_p)
            token_id = next_token.item()
            if token_id == self.config.stop_speech_token:
                break
            generated_ids.append(token_id)
            token = next_token.reshape(1, 1)

            token_emb = self.embed_speech_with_pos(token, offset=step + 1)
            logits, past_kvs = self.forward(token_emb, past_kvs=past_kvs, mask=None)
            logits = logits[:, -1, :].squeeze(0)

        return mx.array(generated_ids, dtype=mx.int64)

    @staticmethod
    def _sample_logits(logits, temperature, repetition_penalty, generated_ids):
        if temperature <= 1e-6:
            temperature = 1e-6
        logits = logits / temperature
        if repetition_penalty != 1.0 and generated_ids:
            for tid in set(generated_ids):
                if logits[tid] < 0:
                    logits[tid] *= repetition_penalty
                else:
                    logits[tid] /= repetition_penalty
        return logits

    @staticmethod
    def _top_p_sample(probs, top_p):
        sorted_indices = mx.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum = mx.cumsum(sorted_probs, axis=-1)
        cutoff = mx.argmax(cumsum >= top_p).item()
        cutoff = max(0, cutoff)
        cutoff_indices = sorted_indices[:cutoff + 1]
        cutoff_sum = mx.sum(sorted_probs[:cutoff + 1])
        cutoff_probs = sorted_probs[:cutoff + 1] / (cutoff_sum + 1e-10)
        sample = mx.random.categorical(mx.log(cutoff_probs + 1e-10), shape=(1,))
        return mx.array([cutoff_indices[sample.item()]], dtype=mx.int32)
