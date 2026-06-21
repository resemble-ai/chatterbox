"""Convert T3 safetensors to MLX-compatible .npz weights."""
import os, sys
from pathlib import Path
import numpy as np
import mlx.core as mx
from safetensors.torch import load_file

KEY_MAP = {}
for i in range(30):
    KEY_MAP[f"tfmr.layers.{i}.self_attn.q_proj.weight"] = f"layers.{i}.attention.q_proj.weight"
    KEY_MAP[f"tfmr.layers.{i}.self_attn.k_proj.weight"] = f"layers.{i}.attention.k_proj.weight"
    KEY_MAP[f"tfmr.layers.{i}.self_attn.v_proj.weight"] = f"layers.{i}.attention.v_proj.weight"
    KEY_MAP[f"tfmr.layers.{i}.self_attn.o_proj.weight"] = f"layers.{i}.attention.o_proj.weight"
    KEY_MAP[f"tfmr.layers.{i}.input_layernorm.weight"] = f"layers.{i}.attention_norm.weight"
    KEY_MAP[f"tfmr.layers.{i}.post_attention_layernorm.weight"] = f"layers.{i}.ffn_norm.weight"
    KEY_MAP[f"tfmr.layers.{i}.mlp.gate_proj.weight"] = f"layers.{i}.mlp.gate.weight"
    KEY_MAP[f"tfmr.layers.{i}.mlp.up_proj.weight"] = f"layers.{i}.mlp.up.weight"
    KEY_MAP[f"tfmr.layers.{i}.mlp.down_proj.weight"] = f"layers.{i}.mlp.down.weight"

KEY_MAP.update({
    "tfmr.norm.weight": "norm.weight",
    "speech_emb.weight": "speech_emb.weight",
    "speech_head.weight": "speech_head.weight",
    "text_emb.weight": "text_emb.weight",
    "text_head.weight": "text_head.weight",
    "speech_pos_emb.emb.weight": "speech_pos_emb.weight",
    "text_pos_emb.emb.weight": "text_pos_emb.weight",
    "cond_enc.spkr_enc.weight": "cond_enc_spkr.weight",
    "cond_enc.spkr_enc.bias": "cond_enc_spkr.bias",
    "cond_enc.emotion_adv_fc.weight": "cond_enc_emotion_fc.weight",
    "cond_enc.perceiver.pre_attention_query": "perceiver_query",
    "cond_enc.perceiver.attn.to_q.weight": "perceiver_to_q.weight",
    "cond_enc.perceiver.attn.to_q.bias": "perceiver_to_q.bias",
    "cond_enc.perceiver.attn.to_k.weight": "perceiver_to_k.weight",
    "cond_enc.perceiver.attn.to_k.bias": "perceiver_to_k.bias",
    "cond_enc.perceiver.attn.to_v.weight": "perceiver_to_v.weight",
    "cond_enc.perceiver.attn.to_v.bias": "perceiver_to_v.bias",
    "cond_enc.perceiver.attn.proj_out.weight": "perceiver_proj_out.weight",
    "cond_enc.perceiver.attn.proj_out.bias": "perceiver_proj_out.bias",
    "cond_enc.perceiver.attn.norm.weight": "perceiver_norm.weight",
    "cond_enc.perceiver.attn.norm.bias": "perceiver_norm.bias",
})


def convert(ckpt_path: str, output_path: str):
    print(f"Loading safetensors from {ckpt_path}")
    sd = load_file(ckpt_path)

    out = {}
    for pt_key, pt_weight in sd.items():
        if pt_key in KEY_MAP:
            mlx_key = KEY_MAP[pt_key]
            out[mlx_key] = mx.array(pt_weight.numpy())
        elif pt_key.startswith("tfmr.layers."):
            if not any(k in pt_key for k in ["rotary", "embed_tokens"]):
                print(f"  WARNING: unmapped backbone key: {pt_key}")
        elif pt_key == "tfmr.embed_tokens.weight":
            pass
        else:
            print(f"  WARNING: unmapped key: {pt_key}")

    print(f"Converted {len(out)} weight tensors")
    print(f"Writing to {output_path}")
    mx.savez(output_path, **out)
    file_size = os.path.getsize(output_path)
    print(f"Saved: {file_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        ckpt_dir = os.path.expanduser("~/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/")
        snapshots = sorted(os.listdir(ckpt_dir))
        if snapshots:
            ckpt_path = os.path.join(ckpt_dir, snapshots[-1], "t3_cfg.safetensors")
        else:
            print("No cached model found. Download it first.")
            sys.exit(1)
    else:
        ckpt_path = sys.argv[1]

    output = sys.argv[2] if len(sys.argv) > 2 else "assets/t3_mlx_weights.npz"

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    convert(ckpt_path, output)
