import argparse
import os

import torch
import torch.nn.functional as F
import torchaudio as ta
from torch.utils.data import Dataset, DataLoader

from diffusers.models.lora import LoRACompatibleLinear, LoRALinearLayer

from chatterbox.tts import ChatterboxTTS
from chatterbox.models.s3tokenizer import S3Tokenizer, S3_SR
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.s3gen.utils.mel import mel_spectrogram
from chatterbox.models.s3gen.utils.mask import make_pad_mask


class TextAudioDataset(Dataset):
    """Simple dataset that loads text and audio pairs from a manifest file."""

    def __init__(self, manifest_path: str, t3_tok, s3_tok: S3Tokenizer, ve, device: str) -> None:
        self.items = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Each line: <text>|<wav_path>
                text, wav = line.split("|", 1)
                self.items.append((text, wav))
        self.t3_tok = t3_tok
        self.s3_tok = s3_tok
        self.ve = ve
        self.device = device

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        text, wav_path = self.items[idx]
        wav, sr = ta.load(wav_path)
        if sr != S3_SR:
            wav = ta.functional.resample(wav, sr, S3_SR)
        wav = wav.squeeze(0)

        speech_tokens, speech_token_len = self.s3_tok(wav.unsqueeze(0))
        mel = mel_spectrogram(wav).transpose(0, 1)  # (T, 80)
        mel_len = torch.tensor([mel.size(0)])

        emb = self.ve.embeds_from_wavs([wav.numpy()], sample_rate=S3_SR)
        emb = torch.from_numpy(emb).float()

        text_tokens = self.t3_tok.text_to_tokens(text).squeeze(0)
        text_tokens = torch.cat([
            torch.tensor([T3Config.start_text_token]),
            text_tokens,
            torch.tensor([T3Config.stop_text_token]),
        ])
        text_token_len = torch.tensor([text_tokens.numel()])

        return {
            "text_tokens": text_tokens,
            "text_token_lens": text_token_len,
            "speech_tokens": speech_tokens.squeeze(0),
            "speech_token_lens": speech_token_len,
            "speech_feat": mel,
            "speech_feat_len": mel_len,
            "embedding": emb.squeeze(0),
        }


def collate_fn(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k.endswith("lens"):
            out[k] = torch.stack([b[k] for b in batch])[:, 0]
        else:
            out[k] = [b[k] for b in batch]

    # Pad sequences
    max_text = max(x.size(0) for x in out["text_tokens"])
    text_pad = []
    for x in out["text_tokens"]:
        text_pad.append(F.pad(x, (0, max_text - x.size(0)), value=T3Config.stop_text_token))
    out["text_tokens"] = torch.stack(text_pad)

    max_speech = max(x.size(0) for x in out["speech_tokens"])
    speech_pad = []
    for x in out["speech_tokens"]:
        speech_pad.append(F.pad(x, (0, max_speech - x.size(0)), value=T3Config.stop_speech_token))
    out["speech_tokens"] = torch.stack(speech_pad)

    max_feat = max(x.size(0) for x in out["speech_feat"])
    feat_pad = []
    for x in out["speech_feat"]:
        feat_pad.append(F.pad(x, (0, 0, 0, max_feat - x.size(0))))
    out["speech_feat"] = torch.stack(feat_pad).transpose(1, 2)  # (B, 80, T)

    out["embedding"] = torch.stack(out["embedding"])
    return out


def attach_lora(model, rank=8):
    for module in model.modules():
        if isinstance(module, LoRACompatibleLinear):
            module.set_lora_layer(LoRALinearLayer(module.in_features, module.out_features, rank))
            module.weight.requires_grad_(False)


def lora_parameters(model):
    params = []
    for module in model.modules():
        if isinstance(module, LoRACompatibleLinear) and module.lora_layer is not None:
            params += list(module.lora_layer.parameters())
    return params


def lora_state_dict(model):
    return {k: v.cpu() for k, v in model.state_dict().items() if v.requires_grad}


def train(args):
    device = args.device
    model = ChatterboxTTS.from_pretrained(device=device)

    attach_lora(model, rank=args.rank)

    t3_tok = model.tokenizer
    s3_tok = S3Tokenizer("speech_tokenizer_v2_25hz")

    ds = TextAudioDataset(args.manifest, t3_tok, s3_tok, model.ve, device)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    opt = torch.optim.AdamW(lora_parameters(model), lr=args.lr)

    os.makedirs(args.outdir, exist_ok=True)

    for epoch in range(args.epochs):
        for batch in dl:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            loss_text, loss_speech = model.t3.loss(
                t3_cond=model.conds.t3,
                text_tokens=batch["text_tokens"],
                text_token_lens=batch["text_token_lens"],
                speech_tokens=batch["speech_tokens"],
                speech_token_lens=batch["speech_token_lens"],
            )

            emb = F.normalize(batch["embedding"], dim=1)
            emb = model.s3gen.flow.spk_embed_affine_layer(emb)
            mask = (~make_pad_mask(batch["speech_token_lens"])).float().unsqueeze(-1).to(device)
            token_embed = model.s3gen.flow.input_embedding(torch.clamp(batch["speech_tokens"], min=0)) * mask
            h, _ = model.s3gen.flow.encoder(token_embed, batch["speech_token_lens"])
            h = model.s3gen.flow.encoder_proj(h)
            h, _ = model.s3gen.flow.length_regulator(h, batch["speech_feat_len"])

            conds = torch.zeros(batch["speech_feat"].shape, device=device)
            for i, j in enumerate(batch["speech_feat_len"]):
                if torch.rand(1) < 0.5:
                    continue
                index = torch.randint(0, int(0.3 * j.item()), (1,))
                conds[i, :, : index] = batch["speech_feat"][i, :, : index]
            mask_feat = (~make_pad_mask(batch["speech_feat_len"])).to(h)
            feat = F.interpolate(batch["speech_feat"].unsqueeze(1), size=h.shape[1:], mode="nearest").squeeze(1)
            loss_flow, _ = model.s3gen.flow.decoder.compute_loss(
                feat.transpose(1, 2).contiguous(),
                mask_feat.unsqueeze(1),
                h.transpose(1, 2).contiguous(),
                emb,
                cond=conds,
            )

            loss = loss_text + loss_speech + loss_flow
            loss.backward()
            opt.step()
            opt.zero_grad()

        ckpt = os.path.join(args.outdir, f"lora_epoch_{epoch}.pt")
        torch.save(lora_state_dict(model), ckpt)
        print(f"saved {ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="txt file with '<text>|<wav>' per line")
    parser.add_argument("--outdir", type=str, default="lora_ckpts")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    train(args)
