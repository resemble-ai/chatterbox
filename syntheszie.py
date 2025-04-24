from time import perf_counter

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


EMO_ADV = 0.5 * torch.ones(1, 1, 1)
device = "cuda"

## T3
states = torch.load("checkpoints/t3.pt")
t3 = T3()
t3.load_state_dict(states)
t3.to(device)
t3.eval()


ref_fp = "tests/trimmed_8b7f38b1.wav"
s3gen_ref_wav, _sr = librosa.load(ref_fp, sr=S3GEN_SR)


## s3gen
s3g_fp = "checkpoints/s3gen.pt"
s3gen = S3Gen()
s3gen.load_state_dict(torch.load(s3g_fp))
s3gen.to(device)
s3gen.eval()
# NOTE: DON'T FORGET TO SET S3GEN TO eval mode! otherwise xvector will trigger error

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
ve = VoiceEncoder()
ve_embed = torch.from_numpy(ve.embeds_from_wavs([s3_ref_wav], sample_rate=S3GEN_SR))
ve_embed = ve_embed.mean(axis=0, keepdim=True).to(device)

t3_cond = T3Cond(
    speaker_emb=ve_embed,
    cond_prompt_speech_tokens=t3_cond_prompt_tokens,
    emotion_adv=EMO_ADV,
).to(device=device)

texts = [
    "Last month, we hit a new milestone.",
    "We reached 2 million views on our YouTube channel, which is absolutely incredible.",
    "Our average watch time is about 5 minutes per user.",
    "If we assume each view lasts for this amount of time, that'll be around 10 billion minutes in total.",
]

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

        print("[ Cosydec inference ]")
        t0_cd = perf_counter()
        wavs, _ = s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=s3gen_ref_dict,
        )

    ta.save(f"syn_out/gen{wavi:02d}.wav", wavs.detach().cpu(), S3GEN_SR)
