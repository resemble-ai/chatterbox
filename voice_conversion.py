from tqdm import tqdm
import sys
import torch
import shutil
from pathlib import Path
import argparse
import os
import librosa
import soundfile as sf
from orator.models.s3tokenizer import S3_SR
from orator.models.s3gen import S3GEN_SR, S3Gen

AUDIO_EXTENSIONS = ["wav", "mp3", "flac", "opus"]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Voice Conversion")
    parser.add_argument("input", type=str, help="Path to input (a sample or folder of samples).")
    parser.add_argument("target_speaker", type=str, help="Path to the sample for the target speaker.")
    parser.add_argument("-o", "--output_folder", type=str, default="vc_outputs")
    parser.add_argument("-g", "--gpu_id", type=int, default=None)
    args = parser.parse_args()

    # Folders
    input = Path(args.input)
    output_folder = Path(args.output_folder)
    output_orig_folder = output_folder / "input"
    output_vc_folder = output_folder / "output"
    ref_folder = output_vc_folder / "target"
    output_orig_folder.mkdir(exist_ok=True, parents=True)
    output_vc_folder.mkdir(exist_ok=True)
    ref_folder.mkdir(exist_ok=True)

    device = torch.device("cpu" if args.gpu_id is None else f"cuda:{args.gpu_id}")

    ## s3gen
    s3g_fp = "checkpoints/s3gen.pt"
    s3gen = S3Gen()
    s3gen.load_state_dict(torch.load(s3g_fp))
    s3gen.to(device)
    s3gen.eval()

    wav_fpaths = []
    if input.is_dir():
        for ext in AUDIO_EXTENSIONS:
            wav_fpaths += list(input.glob(f"*.{ext}"))
    else:
        wav_fpaths.append(input)

    assert wav_fpaths, f"Didn't find any audio in '{input}'"

    ref_24, _ = librosa.load(args.target_speaker, sr=S3GEN_SR, duration=10)
    ref_24 = torch.tensor(ref_24).float()
    shutil.copy(args.target_speaker, ref_folder / Path(args.target_speaker).name)

    for wav_fpath in tqdm(wav_fpaths):
        shutil.copy(wav_fpath, output_orig_folder / wav_fpath.name)

        audio_16, _ = librosa.load(str(wav_fpath), sr=S3_SR)
        audio_16 = torch.tensor(audio_16).float().to(device)[None, ]
        s3_tokens, _ = s3gen.tokenizer(audio_16)

        wav = s3gen(s3_tokens.to(device), ref_24, S3GEN_SR)
        wav = wav.view(-1).cpu().numpy()
        save_path = output_vc_folder / wav_fpath.name
        sf.write(str(save_path), wav, samplerate=S3GEN_SR)


if __name__ == "__main__":
    main()
