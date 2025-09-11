import threading
import torch
import datetime
import functools
import os
import warnings
from pathlib import Path

import psutil
import torchaudio
from loguru import logger
from torch.serialization import safe_globals

from src.chatterbox.models.t3.modules.cond_enc import T3Cond
from src.chatterbox.tts import Conditionals

WAV_OUTPUT_DIR = Path("output_temp")

class ConditionalsCacheManager:
    """Thread-safe cache manager for conditionals with memory and disk storage"""
    
    def __init__(self):
        self._memory_cache = {}
        self._cache_lock = threading.RLock()  # RLock allows re-entrant locking
        self._current_loaded_cache_key = None
        self._disk_save_queue = []  # Track pending disk saves
        
    def get_current_cache_key(self):
        """Get the currently loaded cache key"""
        with self._cache_lock:
            return self._current_loaded_cache_key
    
    def is_cache_key_loaded(self, cache_key):
        """Check if the given cache key is currently loaded"""
        with self._cache_lock:
            return self._current_loaded_cache_key == cache_key
    
    def _save_to_disk_worker(self, cache_key, cond_cls):
        """Non-blocking worker function to save conditionals to disk"""
        try:
            cache_dir = get_cache_dir()
            cache_file = cache_dir.joinpath(cache_key + ".pt")
            arg_dict = dict(
                t3=cond_cls.t3.__dict__,
                gen=cond_cls.gen
            )
            torch.save(arg_dict, cache_file)
            logger.info(f"Saved conditionals to disk cache: {cache_key}")
            
            # Remove from pending queue
            with self._cache_lock:
                if cache_key in self._disk_save_queue:
                    self._disk_save_queue.remove(cache_key)
                    
        except Exception as e:
            logger.error(f"Failed to save conditionals to disk cache: {e}")
            with self._cache_lock:
                if cache_key in self._disk_save_queue:
                    self._disk_save_queue.remove(cache_key)
    
    def save(self, cache_key, cond_cls, enable_memory_cache=True, enable_disk_cache=True):
        """Save prepared conditionals to memory and/or disk cache"""
        if cache_key is None:
            return False
        
        try:
            with self._cache_lock:
                if enable_memory_cache:
                    self._memory_cache[cache_key] = cond_cls.clone()
                    logger.info(f"Saved conditionals object to memory cache: {cache_key}")
                
                self._current_loaded_cache_key = cache_key
                
                # Queue disk save (non-blocking)
                if enable_disk_cache and cache_key not in self._disk_save_queue:
                    self._disk_save_queue.append(cache_key)
                    threading.Thread(
                        target=self._save_to_disk_worker,
                        args=(cache_key, cond_cls),
                        daemon=True,
                        name=f"DiskCacheSave-{cache_key}"
                    ).start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save conditionals cache: {e}")
            return False
    
    def load(self, cache_key, model, device, dtype, enable_memory_cache=True, enable_disk_cache=True):
        """Load prepared conditionals from memory or disk cache"""
        if cache_key is None:
            logger.info("No cache key provided, cannot load conditionals")
            return False
        
        if self.is_cache_key_loaded(cache_key):
            logger.info(f"Conditionals already loaded for cache key: {cache_key}")
            return True
        
        try:
            if enable_memory_cache and self._try_load_from_memory(cache_key, model, device, dtype):
                return True
            
            if enable_disk_cache and self._try_load_from_disk(cache_key, model, device, dtype, enable_memory_cache):
                return True
            logger.info(f"No cached conditionals found for key: {cache_key}")    
            return False
            
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            logger.error(f"Failed to load conditionals cache: {e}")
            return False
    
    def _try_load_from_memory(self, cache_key, model, device, dtype):
        """Try to load conditionals from memory cache with simplified single-device optimization"""
        with self._cache_lock:
            if cache_key in self._memory_cache:
                cached_conditionals = self._memory_cache[cache_key]
                
                model.set_conditionals(cached_conditionals)
                logger.info(f"Loaded conditionals from memory cache: {cache_key}")
                self._current_loaded_cache_key = cache_key
                return True
        return False
    
    
    def _try_load_from_disk(self, cache_key, model, device, dtype, enable_memory_cache, quiet=False):
        """Try to load conditionals from disk cache"""
        cache_dir = get_cache_dir()
        cache_file = cache_dir.joinpath(cache_key + ".pt")
        
        if not cache_file.exists():
            logger.info(f"No disk cache file found for key: {cache_key} in {cache_file}")
            return False
        
        with safe_globals([T3Cond]):
            map_location = torch.device("cuda")
            kwargs = torch.load(cache_file, map_location=map_location, weights_only=True)
            cond_cls = Conditionals(T3Cond(**kwargs['t3']), kwargs['gen'])
        
        if not quiet:
            logger.info(f"Loaded conditionals from disk cache: {cache_key}")

        # Move to target device and ensure correct dtype
        cond_cls = cond_cls.to(device)
        if hasattr(cond_cls.t3, 'speaker_emb') and cond_cls.t3.speaker_emb is not None:
            cond_cls.t3.speaker_emb = cond_cls.t3.speaker_emb.to(dtype=dtype)

        if hasattr(cond_cls, 't3'):
            model.set_conditionals(cond_cls)

        if enable_memory_cache:
            with self._cache_lock:
                self._memory_cache[cache_key] = cond_cls
                if not quiet:
                    logger.info(f"Cached conditionals in memory: {cache_key}")

        with self._cache_lock:
            self._current_loaded_cache_key = cache_key

        return True
    

    def get_cache_stats(self):
        """Get cache statistics"""
        with self._cache_lock:
            memory_keys = list(self._memory_cache.keys())
            
            return {
                'memory_cache_size': len(self._memory_cache),
                'current_loaded_key': self._current_loaded_cache_key,
                'pending_disk_saves': len(self._disk_save_queue),
                'memory_cache_keys': memory_keys
            }


# Global cache manager instance
_cache_manager = ConditionalsCacheManager()


def save_conditionals_cache(cache_key, cond_cls, enable_memory_cache=True, enable_disk_cache=True):
    """Save prepared conditionals to disk (non-blocking) and memory"""
    return _cache_manager.save(cache_key, cond_cls, enable_memory_cache, enable_disk_cache)


def load_conditionals_cache(cache_key, model, device, dtype, enable_memory_cache=True, enable_disk_cache=True):
    """Load prepared conditionals from memory or disk, but only if they've changed"""
    return _cache_manager.load(cache_key, model, device, dtype, enable_memory_cache, enable_disk_cache)


def get_current_cache_key():
    """Get the currently loaded cache key"""
    return _cache_manager.get_current_cache_key()


def is_cache_key_loaded(cache_key):
    """Check if the given cache key is currently loaded"""
    return _cache_manager.is_cache_key_loaded(cache_key)
    

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
def get_cache_key(audio_path, uuid, exaggeration: float = None):
    """Generate a cache key based on audio file, UUID, and exaggeration"""
    if audio_path is None:
        return None

    cache_prefix = Path(audio_path).stem

    # Convert UUID to hex string for readability
    try:
        uuid_hex = hex(uuid)[2:]  # Remove '0x' prefix
    except (TypeError, ValueError):
        uuid_hex = str(uuid)

    if exaggeration is None:
        cache_key = f"{cache_prefix}_{uuid_hex}"
    else:
        cache_key = f"{cache_prefix}_{uuid_hex}_{exaggeration:.2f}"

    return cache_key

@functools.cache
def get_wavout_dir():
    formatted_start_time = get_process_creation_time().strftime("%Y%m%d_%H%M%S")
    wavout_dir = WAV_OUTPUT_DIR.joinpath(formatted_start_time)
    wavout_dir.mkdir(parents=True, exist_ok=True)
    return wavout_dir

def save_torchaudio_wav(wav_tensor, sr, audio_path, uuid):
    """Save a tensor as a WAV file using torchaudio"""

    formatted_now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{formatted_now_time}_{get_cache_key(audio_path, uuid)}"
    path = get_wavout_dir() / f"{filename}.wav"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torchaudio.save(path, wav_tensor.cpu(), sr, encoding="PCM_S")
    return path.resolve()

def init_conditional_memory_cache(model, device, dtype) -> None:
    """Initialize latent cache from disk for all supported languages."""
    latent_dir = get_cache_dir()
    for filename in latent_dir.glob("*.pt"):
        filestem = filename.stem
        if len(filestem.split("_")) < 3:
            continue
        _cache_manager._try_load_from_disk(filestem, model, device, dtype, enable_memory_cache=True, quiet=True)

    stats = _cache_manager.get_cache_stats()

    logger.info(
        f"Loaded conditionals from disk with {stats['memory_cache_size']} entries")

def clear_output_directories():
    """Remove all folders in WAV_OUTPUT_DIR"""
    if not WAV_OUTPUT_DIR.exists():
        logger.info(f"Output directory {WAV_OUTPUT_DIR} does not exist, nothing to clear")
        return 0
    
    removed_count = 0
    try:
        for item in WAV_OUTPUT_DIR.iterdir():
            if item.is_dir():
                import shutil
                shutil.rmtree(item)
                logger.info(f"Removed output directory: {item}")
                removed_count += 1
        
        logger.info(f"Cleared {removed_count} output directories from {WAV_OUTPUT_DIR}")
        return removed_count
        
    except Exception as e:
        logger.error(f"Failed to clear output directories: {e}")
        return 0

def clear_cache_files():
    """Remove all .pt files in the cache directory"""
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        logger.info(f"Cache directory {cache_dir} does not exist, nothing to clear")
        return 0
    
    removed_count = 0
    try:
        for pt_file in cache_dir.glob("*.pt"):
            pt_file.unlink()
            logger.info(f"Removed cache file: {pt_file}")
            removed_count += 1
        
        # Clear memory cache as well since disk files are gone
        global _cache_manager
        with _cache_manager._cache_lock:
            _cache_manager._memory_cache.clear()
            _cache_manager._current_loaded_cache_key = None
            _cache_manager._disk_save_queue.clear()
        
        logger.info(f"Cleared {removed_count} cache files from {cache_dir}")
        logger.info("Cleared memory cache as well")
        return removed_count
        
    except Exception as e:
        logger.error(f"Failed to clear cache files: {e}")
        return 0
