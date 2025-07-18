import functools
import hashlib
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr
import numpy as np
import torch

from src.InterruptionFlag import interruptible, InterruptionFlag
from src.chatterbox.models.t3.modules.cond_enc import T3Cond
from src.chatterbox.tts import Conditionals
from src.simple_model_state import simple_manage_model_state

if TYPE_CHECKING:
    from src.chatterbox.tts import ChatterboxTTS


def split_by_lines(prompt: str):
    prompts = re.split(r'(?<=[.?!])\s*(?![.\w"\'\d]|[,!]|\*)', prompt)
    prompts = [p.strip() for p in prompts if p.strip()]
    return prompts

def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def resolve_device(device):
    return get_best_device() if device == "auto" else device


def resolve_dtype(dtype):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


def t3_to(model: "ChatterboxTTS", dtype):
    model.t3.to(dtype=dtype)
    model.conds.t3.to(dtype=dtype)
    return model


def s3gen_to(model: "ChatterboxTTS", dtype):
    if dtype == torch.float16:
        model.s3gen.flow.fp16 = True
    elif dtype == torch.float32:
        model.s3gen.flow.fp16 = False
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")
    # model.s3gen.flow.to(dtype=dtype)
    model.s3gen.to(dtype=dtype)
    model.s3gen.mel2wav.to(dtype=torch.float32)
    # due to "Error: cuFFT doesn't support tensor of type: BFloat16" from torch.stft
    # and other errors and general instability
    model.s3gen.tokenizer.to(dtype=torch.float32)
    model.s3gen.speaker_encoder.to(dtype=torch.float32)
    return model


def chatterbox_tts_to(model: "ChatterboxTTS", device, dtype):
    print(f"Moving model to {str(device)}, {str(dtype)}")

    model.ve.to(device=device)
    # model.conds.to(device=device)
    # model.t3.to(device=device, dtype=dtype)
    t3_to(model, dtype)
    # model.s3gen.to(device=device, dtype=dtype)
    # # due to "Error: cuFFT doesn't support tensor of type: BFloat16" from torch.stft
    # model.s3gen.tokenizer.to(dtype=torch.float32)
    s3gen_to(model, dtype if dtype != torch.bfloat16 else torch.float16)
    model.device = device
    torch.cuda.empty_cache()

    return model


def _set_t3_compilation(model: "ChatterboxTTS"):
    if not hasattr(model.t3, "_step_compilation_target_original"):
        model.t3._step_compilation_target_original = model.t3._step_compilation_target
    model.t3._step_compilation_target = torch.compile(
        model.t3._step_compilation_target, fullgraph=True, backend="cudagraphs"
    )


def compile_t3(model: "ChatterboxTTS"):
    _set_t3_compilation(model)
    for i in range(2):
        print(f"Compiling T3 {i + 1}/2")
        list(model.generate("triggering torch compile by running the model"))


def remove_t3_compilation(model: "ChatterboxTTS"):
    if not hasattr(model.t3, "_step_compilation_target_original"):
        return
    model.t3._step_compilation_target = model.t3._step_compilation_target_original


@simple_manage_model_state("chatterbox")
def get_model(model_name="just_a_placeholder",
    device=torch.device("cuda"), dtype=torch.float32
):
    from src.chatterbox.tts import ChatterboxTTS

    model = ChatterboxTTS.from_pretrained(device=device)
    # having everything on float32 increases performance
    return chatterbox_tts_to(model, device, dtype)


def generate_model_name(device, dtype):
    return f"Chatterbox on {device} with {dtype}"


@contextmanager
def chatterbox_model(model_name, device="cuda", dtype=torch.float32):
    model = get_model(
        model_name=generate_model_name(device, dtype),
        device=torch.device(device),
        dtype=dtype,
    )

    # use_autocast = dtype in [torch.float16, torch.bfloat16]

    # with (
    #     torch.autocast(device_type=device, dtype=dtype)
    #     if use_autocast
    #     else torch.no_grad()
    # ):
    with torch.no_grad():
        yield model


@contextmanager
def cpu_offload_context(model, device, dtype, cpu_offload=False):
    if cpu_offload:
        chatterbox_tts_to(model, torch.device(device), dtype)
    yield model
    if cpu_offload:
        chatterbox_tts_to(model, torch.device("cpu"), dtype)


# Global in-memory cache for conditionals
_conditionals_memory_cache = {}
_cache_lock = threading.Lock()

def get_cache_dir():
    """Get or create the conditionals cache directory"""
    cache_dir = Path("cache/conditionals")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def get_cache_key(audio_path, uuid, exaggeration):
    """Generate a cache key based on audio file, UUID, and exaggeration"""
    if audio_path is None:
        return None

    # Extract just the filename without extension as prefix
    try:
        filename = Path(audio_path).stem  # Gets filename without extension
        # Remove any temp directory prefixes, just keep the actual filename
        cache_prefix = filename
    except Exception:
        cache_prefix = "unknown"

    # Convert UUID to hex string for readability
    try:
        uuid_hex = hex(uuid)[2:]  # Remove '0x' prefix
    except (TypeError, ValueError):
        uuid_hex = str(uuid)

    # Create cache key: prefix_uuid_exaggeration
    cache_key = f"{cache_prefix}_{uuid_hex}_{exaggeration}"

    # Use MD5 hash if the key gets too long (over 100 chars)
    if len(cache_key) > 100:
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return f"{cache_prefix}_{cache_hash}"

    return cache_key

def _save_conditionals_to_disk(cache_key, cond_cls):
    """Non-blocking worker function to save conditionals to disk"""
    try:
        cache_dir = get_cache_dir()
        cache_file = cache_dir.joinpath(cache_key + ".pt")
        arg_dict = dict(
            t3=cond_cls.t3.__dict__,
            gen=cond_cls.gen
        )
        torch.save(arg_dict, cache_file)

        print(f"Saved conditionals cache: {cache_key}")

    except Exception as e:
        print(f"Failed to save conditionals cache: {e}")

def save_conditionals_cache(cache_key, cond_cls):
    """Save prepared conditionals to disk (non-blocking) and memory"""
    if cache_key is None:
        return

    # TODO Disk cache possibly broken, not used
    #try:
    #    # Save to memory cache (blocking, but fast)
    #    with _cache_lock:
    #        _conditionals_memory_cache[cache_key] = {'t3':cond_cls.t3, 'gen':cond_cls.gen}
    #        print(f"Saved conditionals to memory cache: {cache_key}")

    #    # Save to disk (non-blocking)
    #    threading.Thread(
    #        target=_save_conditionals_to_disk,
    #        args=(cache_key, cond_cls),
    #        daemon=True
    #    ).start()

    #except Exception as e:
    #    print(f"Failed to prepare conditionals cache: {e}")

def load_conditionals_cache(cache_key, model, device, dtype):
    """Load prepared conditionals from memory or disk with optional GDS acceleration"""
    if cache_key is None:
        return None

    try:
        # Try memory cache first (fastest)
        with _cache_lock:
            if cache_key in _conditionals_memory_cache:
                cache_data = _conditionals_memory_cache[cache_key]

                # Restore conditionals to model
                if 't3' in cache_data and cache_data['t3'] is not None:
                    model.set_conditionals(Conditionals(T3Cond(cache_data['t3']), cache_data['gen']))

                print(f"Loaded conditionals from memory cache: {cache_key}")
                return True

        # TODO Disk cache possibly broken, not used
        ## Try disk cache with GDS
        #cache_dir = get_cache_dir()
        #cache_file = cache_dir.joinpath(cache_key + ".pt")
#
        #if not cache_file.exists():
        #    return None
#
        #with safe_globals([T3Cond]):
        #    #cond_cls = Conditionals.load(cls=Conditionals,fpath=cache_file)
        #    map_location = torch.device("cpu")
        #    kwargs = torch.load(cache_file, map_location=map_location, weights_only=True)
        #    cond_cls =  Conditionals(T3Cond(**kwargs['t3']), kwargs['gen'])
        #print(f"loaded cond {cond_cls.__sizeof__()}")
        ## Restore conditionals to model
        #if hasattr(cond_cls, 't3'):
        #   model.set_conditionals(cond_cls)
        #   print(f"set conditionals")
#
        ## Store in memory cache for next time
        #cache_dict = dict(
        #    t3=cond_cls.__dict__,
        #    gen=cond_cls.gen
        #)
        #with _cache_lock:
        #    _conditionals_memory_cache[cache_key] = cache_dict
#
        #print(f"Loaded conditionals cache: {cache_key}")
        #return True

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print(f"Failed to load conditionals cache: {e}")
        return None

@interruptible
def _tts_generator(
    text,
    exaggeration=0.5,
    cfgw=0.5,
    temperature=0.8,
    audio_prompt_path=None,
    # model
    model_name="just_a_placeholder",
    device="cuda",
    dtype="float32",
    cpu_offload=False,
    # hyperparameters
    chunked=False,
    cache_voice=False,
    # streaming
    #tokens_per_slice=1000,
    #remove_milliseconds=100,
    #remove_milliseconds_start=100,
    #chunk_overlap_method="zero",
    # chunks
    desired_length=200,
    max_length=300,
    halve_first_chunk=False,
    seed_num=-1,
    progress=gr.Progress(),
    streaming=False,
    use_compilation=None,
    max_new_tokens=1000,
    max_cache_len=1500,
    # New parameter for disk caching
    cache_uuid=None,
    # Additional parameters
    min_p=0.05,
    top_p=1.0,
    repetition_penalty=1.2,
    do_progress=True,
    **kwargs,
):
    device = resolve_device(device)
    dtype = resolve_dtype(dtype)

    print(f"Using device: {device}")

    progress(0.0, desc="Retrieving model...")
    with (chatterbox_model(
        model_name=model_name,
        device=device,
        dtype=dtype,
    ) as model, cpu_offload_context(model, device, dtype, cpu_offload)):
        progress(0.1, desc="Generating audio...")

        if use_compilation:
            _set_t3_compilation(model)
        else:
            remove_t3_compilation(model)

        # Enhanced conditional preparation with disk caching
        if audio_prompt_path is not None:
            # Generate cache key
            cache_key = get_cache_key(audio_prompt_path, cache_uuid, exaggeration)
            conditionals_loaded = False

            # Try to load from disk cache first
            if cache_key and load_conditionals_cache(cache_key, model, device, dtype):
                conditionals_loaded = True
                progress(0.3, desc="Loaded cached conditionals...")

            # If not loaded from cache, prepare and optionally cache
            if not conditionals_loaded:
                progress(0.2, desc="Preparing conditionals...")
                model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)

                if dtype != torch.float32:
                    model.conds.t3.to(dtype=dtype)

                # Save to disk cache if we have a cache key
                if cache_key:
                    save_conditionals_cache(cache_key, model.conds)

            # Update in-memory cache tracking
            if cache_voice:
                model._cached_prompt_path = audio_prompt_path

        def generate_chunk(text):
            print(f"Generating chunk: {text}")
            yield from model.generate(
                text,
                exaggeration=exaggeration,
                cfg_weight=cfgw,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                max_cache_len=max_cache_len,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )

        texts = (
            split_by_lines(text)
            if chunked
            else [text]
        )

        for i, chunk in enumerate(texts):
            if not streaming:
               progress(i / len(texts), desc=f"Generating chunk: {chunk}")

            chunk_wavs = list(generate_chunk(chunk))

            if chunk_wavs:
                with torch.no_grad():
                    if len(chunk_wavs) == 1:
                        wav_tensor = chunk_wavs[0].squeeze()
                        # Yield both for streaming and for final concatenation
                        yield {"audio_out": (model.sr, wav_tensor.cpu().numpy()), "wav_tensor": wav_tensor}
                    else:
                        stacked_wavs = torch.stack(chunk_wavs).squeeze()

                        if stacked_wavs.ndim == 1:
                            yield {"audio_out": (model.sr, stacked_wavs.cpu().numpy()), "wav_tensor": stacked_wavs}
                        else:
                            for wav_tensor in stacked_wavs:
                                yield {"audio_out": (model.sr, wav_tensor.cpu().numpy()), "wav_tensor": wav_tensor}

        # Signal completion with device info for GPU optimization
        yield {"device": device, "sr": model.sr}


global_interrupt_flag = InterruptionFlag()


@functools.wraps(_tts_generator)
def tts(*args, **kwargs):
    start_time = time.time()
    try:
        # Collect all results efficiently
        results = list(
            _tts_generator(*args, interrupt_flag=global_interrupt_flag, **kwargs)
        )
        if not results:
            raise gr.Error("No audio generated")

        # Extract wav tensors and metadata
        wav_tensors = []
        sr = None
        device = "cpu"

        for result in results:
            if "wav_tensor" in result:
                wav_tensors.append(result["wav_tensor"])
                if sr is None:
                    sr = result["audio_out"][0]
            elif "device" in result:
                device = result["device"]
                if sr is None:
                    sr = result["sr"]

        if not wav_tensors:
            raise gr.Error("No audio generated")

        # Determine if we should use GPU optimization
        use_gpu = (device != "cpu" and
                  len(wav_tensors) > 0 and
                  torch.is_tensor(wav_tensors[0]) and
                  wav_tensors[0].is_cuda)
        # Optimized concatenation based on device
        with torch.no_grad():
            if use_gpu and len(wav_tensors) > 1:
                # GPU-based concatenation - keep tensors on GPU until final conversion
                full_wav_tensor = torch.cat(wav_tensors, dim=0)
                full_wav = full_wav_tensor.cpu().numpy()
            elif use_gpu and len(wav_tensors) == 1:
                # Single tensor on GPU
                full_wav = wav_tensors[0].cpu().numpy()
            else:
                # CPU fallback - convert tensors to numpy first, then concatenate
                wav_arrays = [t.cpu().numpy() if torch.is_tensor(t) else t for t in wav_tensors]
                full_wav = np.concatenate(wav_arrays, axis=0) if len(wav_arrays) > 1 else wav_arrays[0]

        # Calculate timing and performance metrics
        end_time = time.time()
        execution_time = end_time - start_time
        audio_length_seconds = len(full_wav) / sr
        speed_ratio = audio_length_seconds / execution_time

        print(f"  Execution time: {execution_time:.2f}s  Audio length: {audio_length_seconds:.2f}s")
        print(f"  Speed ratio: {speed_ratio:.2f}x (audio_length/execution_time)")

        return sr, full_wav

    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"TTS Generation Failed after {execution_time:.2f}s")
        import traceback
        print(traceback.format_exc())
        raise gr.Error(f"Error: {e}")
