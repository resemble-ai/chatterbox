import threading
import torch
import datetime
import functools
import hashlib
import os
import time
import warnings
from pathlib import Path

import psutil
import torchaudio
from torch.serialization import safe_globals

from src.chatterbox.models.t3.modules.cond_enc import T3Cond
from src.chatterbox.tts import Conditionals

# Global in-memory cache for conditionals
_conditionals_memory_cache = {}
_cache_lock = threading.Lock()
_current_loaded_cache_key = None  # Track currently loaded conditionals


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

def save_conditionals_cache(cache_key, cond_cls, enable_memory_cache=True, enable_disk_cache=True):
    """Save prepared conditionals to disk (non-blocking) and memory"""
    if cache_key is None:
        return

    try:
        # Save to memory cache (blocking, but fast)
        if enable_memory_cache:
            with _cache_lock:
                _conditionals_memory_cache[cache_key] = {
                    't3_dict': cond_cls.t3.__dict__.copy(),
                    'gen': cond_cls.gen
                }
                print(f"Saved conditionals to memory cache: {cache_key}")

        # Save to disk (non-blocking)
        if enable_disk_cache:
            threading.Thread(
                target=_save_conditionals_to_disk,
                args=(cache_key, cond_cls),
                daemon=True
            ).start()

    except Exception as e:
        print(f"Failed to prepare conditionals cache: {e}")


def load_conditionals_cache(cache_key, model, device, dtype, enable_memory_cache=True, enable_disk_cache=True):
    """Load prepared conditionals from memory or disk, but only if they've changed"""
    global _current_loaded_cache_key

    if cache_key is None:
        return None

    # Check if we already have the right conditionals loaded
    if _current_loaded_cache_key == cache_key:
        print(f"Conditionals already loaded for cache key: {cache_key}")
        return True

    try:
        # Try memory cache first (fastest)
        if enable_memory_cache:
            with _cache_lock:
                if cache_key in _conditionals_memory_cache:
                    cache_data = _conditionals_memory_cache[cache_key]

                    # Restore conditionals from dict
                    if 't3_dict' in cache_data and 'gen' in cache_data:
                        # Recreate T3Cond from dict
                        t3_cond = T3Cond(**cache_data['t3_dict'])
                        t3_cond = t3_cond.to(device=device, dtype=dtype)

                        # Create new Conditionals object
                        conditionals = Conditionals(t3_cond, cache_data['gen'])
                        model.set_conditionals(conditionals)

                    print(f"Loaded conditionals from memory cache: {cache_key}")
                    _current_loaded_cache_key = cache_key  # Update tracking
                    return True

        # Try disk cache if memory cache missed or is disabled
        if enable_disk_cache:
            cache_dir = get_cache_dir()
            cache_file = cache_dir.joinpath(cache_key + ".pt")

            if not cache_file.exists():
                return None

            with safe_globals([T3Cond]):
                #cond_cls = Conditionals.load(cls=Conditionals,fpath=cache_file)
                map_location = torch.device("cuda")
                kwargs = torch.load(cache_file, map_location=map_location, weights_only=True)
                cond_cls =  Conditionals(T3Cond(**kwargs['t3']), kwargs['gen'])
            print(f"loaded cond {cond_cls.__sizeof__()}")
            # Restore conditionals to model
            if hasattr(cond_cls, 't3'):
               model.set_conditionals(cond_cls)
               print(f"set conditionals")

            # Store in memory cache for next time (if memory cache is enabled)
            if enable_memory_cache:
                cache_dict = dict(
                    t3=cond_cls.__dict__,
                    gen=cond_cls.gen
                )
                with _cache_lock:
                    _conditionals_memory_cache[cache_key] = cache_dict

            print(f"Loaded conditionals cache: {cache_key}")
            _current_loaded_cache_key = cache_key  # Update tracking
            return True

        return None

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print(f"Failed to load conditionals cache: {e}")
        return None
    
@functools.cache
def get_process_creation_time():
    """Get the process creation time as a datetime object"""
    p = psutil.Process(os.getpid())
    creation_timestamp = p.create_time()
    return datetime.datetime.fromtimestamp(creation_timestamp)

@functools.cache
def get_cache_dir():
    """Get or create the conditionals cache directory"""
    cache_dir = Path("cache").joinpath("conditionals")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

@functools.cache
def get_cache_key(audio_path, uuid, exaggeration=None):
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
    if exaggeration is None:
        cache_key = f"{cache_prefix}_{uuid_hex}"
    else:
        cache_key = f"{cache_prefix}_{uuid_hex}_{exaggeration:.2f}"

    # Use MD5 hash if the key gets too long (over 100 chars)
    if len(cache_key) > 100:
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return f"{cache_prefix}_{cache_hash}"

    return cache_key

@functools.cache
def get_wavout_dir():
    formatted_start_time = get_process_creation_time().strftime("%Y%m%d_%H%M%S")
    wavout_dir = Path("output_temp").joinpath(formatted_start_time)
    wavout_dir.mkdir(parents=True, exist_ok=True)
    return wavout_dir

def save_torchaudio_wav(wav_tensor, sr, audio_path, uuid):
    """Save a tensor as a WAV file using torchaudio"""

    formatted_now_time = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")

    filename = f"{formatted_now_time}_{get_cache_key(audio_path, uuid)}"
    path = get_wavout_dir() / f"{filename}.wav"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        torchaudio.save(path, wav_tensor.to("cpu"), sr, encoding="PCM_S", bits_per_sample=16)
    return path.resolve()