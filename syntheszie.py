from pathlib import Path
from time import perf_counter
from argparse import ArgumentParser

import librosa
import torch
import torch.nn.functional as F
import torchaudio as ta

from models.t3 import T3
from models.s3tokenizer import S3_SR, drop_invalid_tokens
from models.s3gen import S3GEN_SR, S3Gen
from models.tokenizers import EnTokenizer
from models.voice_encoder import VoiceEncoder
from models.t3.modules.cond_enc import T3Cond


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-tf", "--text_file", type=Path, default=None, required=True, help="Path to the text file")
    parser.add_argument("-o", "--out_dir", type=Path, default=Path("syn_out"))
    parser.add_argument("-sp", "--speech_prompt", type=Path, default=None, required=True, help="Path to the speech prompt wav file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    parser.add_argument("-e", "--emotion_adv", type=float, default=0.5, help="Emotion ADV weight")

    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True, parents=True)
    return args


def synthesize(
    out_dir,
    text_file,
    speech_prompt,
    device,
    emotion_adv,
):
    ## T3
    states = torch.load("checkpoints/t3.pt")
    t3 = T3()
    t3.load_state_dict(states)
    t3.to(device)
    t3.eval()

    ## VE
    states = torch.load("checkpoints/ve.pt")
    ve = VoiceEncoder()
    ve.load_state_dict(states)
    ve.to(device)
    ve.eval()

    ## s3gen
    s3g_fp = "checkpoints/s3gen.pt"
    s3gen = S3Gen()
    s3gen.load_state_dict(torch.load(s3g_fp))
    s3gen.to(device)
    s3gen.eval()
    # NOTE: DON'T FORGET TO SET S3GEN TO eval mode! otherwise xvector will trigger error


    ## Load reference wav
    s3gen_ref_wav, _sr = librosa.load(speech_prompt, sr=S3GEN_SR)

    s3_ref_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)


    s3gen_ref_wav = s3gen_ref_wav[:10 * S3GEN_SR]
    s3gen_ref_dict = s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=device)


    # Speech cond prompt tokens
    if plen := t3.hp.speech_cond_prompt_len:
        s3_tokzr = s3gen.tokenizer
        t3_cond_prompt_tokens, _ = s3_tokzr.forward([s3_ref_wav], max_len=plen)
        t3_cond_prompt_tokens = t3_cond_prompt_tokens[0, :t3.hp.speech_cond_prompt_len]
        t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(device)


    # # Voice-encoder speaker embedding
    ve_embed = torch.from_numpy(ve.embeds_from_wavs([s3_ref_wav], sample_rate=S3GEN_SR))
    ve_embed = ve_embed.mean(axis=0, keepdim=True).to(device)

    t3_cond = T3Cond(
        speaker_emb=ve_embed,
        cond_prompt_speech_tokens=t3_cond_prompt_tokens,
        emotion_adv=emotion_adv * torch.ones(1, 1, 1),
    ).to(device=device)


    ## Synthesis
    texts = text_file.read_text().splitlines()
    for wavi, text in enumerate(texts):
        tokenizer = EnTokenizer("checkpoints/tokenizer.json")
        text_tokens = tokenizer.text_to_tokens(text).to(device)

        t_start, t_stop = t3.hp.start_text_token, t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=t_start)
        text_tokens = F.pad(text_tokens, (0, 1), value=t_stop)

        with torch.inference_mode():
            print("[ HF inference ]")
            t0_hf = perf_counter()

            speech_tokens = t3.inference(
                t3_cond=t3_cond,
                text_tokens=text_tokens,
                max_new_tokens=1000,
            )

            t1_hf = perf_counter()
            print(f"> took {int(1000*(t1_hf - t0_hf))} ms")
            print(f"> hf tokens:", speech_tokens.shape)

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(t3.device)

            print("[ S3Gen inference ]")
            t0_cd = perf_counter()
            wavs, _ = s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=s3gen_ref_dict,
            )

        ta.save(out_dir / f"gen{wavi:02d}.wav", wavs.detach().cpu(), S3GEN_SR)

if __name__ == "__main__":
    args = parse_args()
    arg_dict = vars(args)
    synthesize(**arg_dict)
