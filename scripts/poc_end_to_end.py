"""End-to-end PoC: PyTorch cond + MLX T3 generate + S3Gen decode."""
import sys, os
import numpy as np
import torch
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from chatterbox.models.t3.t3 import T3, T3Config
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.s3gen import S3Gen
from safetensors.torch import load_file
from chatterbox.models.tokenizers import EnTokenizer
from chatterbox.models.t3_mlx import T3MLX, T3MLXConfig, _BACKBONE_KEYS

CACHE = os.path.expanduser(
    "~/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
)
CKPT = os.path.join(CACHE, "t3_cfg.safetensors")


def load_pt_t3(device="cpu"):
    hp = T3Config.english_only()
    model = T3(hp).to(device).eval()
    sd = load_file(CKPT)
    tfmr_sd = {k[5:]: v for k, v in sd.items() if k.startswith("tfmr.") and "embed_tokens" not in k}
    model.tfmr.load_state_dict(tfmr_sd, strict=False)
    model.speech_emb.load_state_dict({"weight": sd["speech_emb.weight"]})
    model.speech_head.load_state_dict({"weight": sd["speech_head.weight"]})
    model.text_emb.load_state_dict({"weight": sd["text_emb.weight"]})
    model.text_head.load_state_dict({"weight": sd["text_head.weight"]})
    model.text_pos_emb.load_state_dict({"emb.weight": sd["text_pos_emb.emb.weight"]})
    model.speech_pos_emb.load_state_dict({"emb.weight": sd["speech_pos_emb.emb.weight"]})
    cond_sd = {k[9:]: v for k, v in sd.items() if k.startswith("cond_enc.")}
    model.cond_enc.load_state_dict(cond_sd, strict=False)
    print("PyTorch T3 loaded")
    return model


def get_cond_emb(pt_model, spkr_emb_np, prompt_tokens_np, emotion_adv=0.5):
    device = next(pt_model.parameters()).device
    spkr_emb = torch.from_numpy(spkr_emb_np).to(device)
    prompt_tokens = torch.from_numpy(prompt_tokens_np).long().to(device)
    emotion = torch.tensor([[[emotion_adv]]], dtype=torch.float, device=device)
    # Pass raw tokens, prepare_conditioning will embed + add position emb
    t3_cond = T3Cond(
        speaker_emb=spkr_emb,
        cond_prompt_speech_tokens=prompt_tokens,
        cond_prompt_speech_emb=None,
        emotion_adv=emotion,
    )
    with torch.inference_mode():
        cond_emb = pt_model.prepare_conditioning(t3_cond)
    return cond_emb


if __name__ == "__main__":
    device = "cpu"
    pt_model = load_pt_t3(device)
    s3gen = S3Gen(meanflow=False).to(device).eval()
    s3_sd = load_file(os.path.join(CACHE, "s3gen.safetensors"))
    s3gen.load_state_dict(s3_sd, strict=False)
    print("S3Gen loaded")

    # Dummy conditioning for PoC (replace with real audio later)
    rng = np.random.RandomState(42)
    spkr_emb = rng.randn(1, 256).astype(np.float32)
    prompt_tokens = rng.randint(0, 8194, size=(1, 150)).astype(np.int64)
    cond_emb = get_cond_emb(pt_model, spkr_emb, prompt_tokens)

    # Tokenize text
    text = "Hello, this is a test of the text to speech system."
    tokenizer = EnTokenizer(os.path.join(CACHE, "tokenizer.json"))
    toks = tokenizer.text_to_tokens(text).to(device)
    sot = torch.tensor([[pt_model.hp.start_text_token]], device=device)
    eot = torch.tensor([[pt_model.hp.stop_text_token]], device=device)
    text_tokens = torch.cat([sot, toks, eot], dim=1)
    print(f"Text: '{text}' -> {text_tokens.shape[1]} tokens")

    # MLX generate
    cfg = T3MLXConfig()
    mlx_model = T3MLX(cfg)
    w = mx.load("assets/t3_mlx_weights.npz")
    backbone = [(k, v) for k, v in w.items() if k in _BACKBONE_KEYS]
    mlx_model.load_weights(backbone)
    print("MLX T3 loaded")

    speech_tokens_mx = mlx_model.generate(
        mx.array(cond_emb.cpu().numpy()),
        mx.array(text_tokens.cpu().numpy(), dtype=mx.int32),
        max_new_tokens=500, seed=42,
    )
    print(f"Generated {len(speech_tokens_mx)} tokens")

    # S3Gen decode
    speech_tokens_pt = torch.from_numpy(np.array(speech_tokens_mx)).long().to(device)[None]

    # Dummy reference for S3Gen
    ref_wav = torch.zeros(1, 24000 * 10)
    with torch.inference_mode():
        wav, _ = s3gen.inference(speech_tokens=speech_tokens_pt, ref_wav=ref_wav, ref_sr=24000)

    import soundfile as sf
    out_path = "output_poc_mlx.wav"
    sf.write(out_path, wav[0].cpu().numpy(), 24000)
    print(f"Saved {out_path} ({len(wav[0]) / 24000:.1f}s)")
