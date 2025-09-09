import os
import random
import shutil
import numpy as np
import torch

from sys import (stdout)
from time import  perf_counter_ns
from src.cache_utils import (
    load_conditionals_cache,
   save_conditionals_cache,
   get_cache_key,
   save_torchaudio_wav
)
# Third-party imports
from pathlib import Path

import torch
from loguru import logger
from src.chatterbox.tts import ChatterboxTTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model():
    model = ChatterboxTTS.from_pretrained(DEVICE)
    model.t3.to(dtype=DTYPE)
    model.conds.t3.to(dtype=DTYPE)
    torch.cuda.empty_cache()
    return model


def generate(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw):
    enable_memory_cache = True
    enable_disk_cache = True
    cache_voice = True
    cache_uuid = seed_num
    device = DEVICE
    dtype = DTYPE

    if model is None:
        model = load_model()

    if seed_num != 0:
        set_seed(int(seed_num))

    func_start_time = perf_counter_ns()

    # Enhanced conditional preparation with configurable caching
    if audio_prompt_path is not None:
        # Generate cache key
        cache_key = get_cache_key(audio_prompt_path, cache_uuid, exaggeration)
        conditionals_loaded = False
        # Try to load from cache first (respecting cache flags)
        if cache_key and (enable_memory_cache or enable_disk_cache):
            if load_conditionals_cache(cache_key, model, device, dtype, enable_memory_cache, enable_disk_cache):
                conditionals_loaded = True
        # If not loaded from cache, prepare and optionally cache
        if not conditionals_loaded:
            model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
            if dtype != torch.float32:
                model.conds.t3.to(dtype=dtype)
            # Save to cache if we have a cache key and caching is enabled
            if cache_key and (enable_memory_cache or enable_disk_cache):
                save_conditionals_cache(cache_key, model.conds, enable_memory_cache, enable_disk_cache)
        # Update in-memory cache tracking
        if cache_voice:
            model._cached_prompt_path = audio_prompt_path
    conditional_start_time = perf_counter_ns()
    logger.info(f"Conditionals prepared. Time: {(conditional_start_time - func_start_time) / 1_000_000:.4f}ms")
    generate_start_time = perf_counter_ns()
    wav = model.generate(
        text,
        #audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        temperature=temperature,
        cfg_weight=cfgw,
        t3_params={
            #"initial_forward_pass_backend": "eager", # slower - default
            #"initial_forward_pass_backend": "cudagraphs", # speeds up set up
            "generate_token_backend": "cudagraphs-manual", # fastest - default
            #"generate_token_backend": "cudagraphs",
            # "generate_token_backend": "eager",
            # "generate_token_backend": "inductor",
            # "generate_token_backend": "inductor-strided",
            #"generate_token_backend": "cudagraphs-strided",
            #"stride_length": 4, # "strided" options compile <1-2-3-4> iteration steps together, which improves performance by reducing memory copying issues in torch.compile
            "skip_when_1": True, # skips Top P when it's set to 1.0
            #"benchmark_t3": True, # Synchronizes CUDA to get the real it/s 
        }
    )
    #logger.info(f"Generation completed. Time: {(perf_counter_ns() - generate_start_time) / 1_000_000_000:.2f}s")
    # Log execution time
    func_end_time = perf_counter_ns()

    total_duration_s = (func_end_time - func_start_time)  / 1_000_000_000  # Convert nanoseconds to seconds
    wav_length = wav.shape[-1]   / model.sr

    logger.info(f"Generated audio: {wav_length:.2f}s {model.sr/1000:.2f}kHz in {total_duration_s:.2f}s. Speed: {wav_length / total_duration_s:.2f}x")
    wave_file = str(save_torchaudio_wav(wav.cpu(), model.sr, audio_path=audio_prompt_path, uuid=cache_uuid))
    del wav
    torch.cuda.empty_cache()
    return None #wave_file
    #return (model.sr, wav.squeeze(0).cpu().numpy())


if __name__ == "__main__":
    #shutil.rmtree(Path("cache").joinpath("conditionals"), ignore_errors=True)
    #test_text= "Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible."
    test_text= "Now let's make my mum's favourite. Oh boy this is going to be incredible."
    test_asset2=Path.cwd().joinpath("assets", "dlc1seranavoice.wav")
    test_asset = Path.cwd().joinpath("assets", "fishaudio_horror.wav")
    model = load_model()
    #wavfile = generate(model, test_text, None, exaggeration=0.65, temperature=0.8, seed_num=420, cfgw=0)
    wavfile = generate(model, test_text, test_asset, exaggeration=0.65, temperature=0.9, seed_num=420, cfgw=0)
    wavfile = generate(model, test_text, test_asset2, exaggeration=0.65, temperature=0.9, seed_num=420, cfgw=0)
    wavfile = generate(model, test_text, test_asset, exaggeration=0.65, temperature=0.9, seed_num=420, cfgw=0)
    wavfile = generate(model, test_text, test_asset, exaggeration=0.65, temperature=0.9, seed_num=420, cfgw=0)
    wavfile = generate(model, test_text, test_asset2, exaggeration=0.65, temperature=0.9, seed_num=420, cfgw=0)
    #print(f"Generated wav file: {wavfile}")

