import threading
import torch
import datetime
import functools
import os
import warnings
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Dict, Any
import gc

import psutil
import torchaudio
from loguru import logger
from torch.serialization import safe_globals

from src.chatterbox.models.t3.modules.cond_enc import T3Cond
from src.chatterbox.tts import Conditionals

WAV_OUTPUT_DIR = Path("output_temp")


class LRUCache:
    """Thread-safe LRU cache with size limits and memory management."""
    
    def __init__(self, max_size: int = 10, max_memory_mb: int = 512, lightweight: bool = False):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.lightweight = lightweight  # Skip expensive memory estimation if True
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache and move to end (most recently used)."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                value = self._cache.pop(key)
                self._cache[key] = value
                return value
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache, evicting old items if necessary."""
        with self._lock:
            # If key exists, update and move to end
            if key in self._cache:
                self._cache.pop(key)
            
            self._cache[key] = value
            
            # Evict if necessary
            self._evict_if_needed()
    
    def _evict_if_needed(self) -> None:
        """Evict least recently used items if cache exceeds limits."""
        # Size-based eviction (always enabled)
        while len(self._cache) > self.max_size:
            oldest_key, oldest_value = self._cache.popitem(last=False)
            logger.info(f"LRU cache evicted by size: {oldest_key}")
            self._cleanup_value(oldest_value)
        
        # Memory-based eviction (skip if lightweight mode)
        if not self.lightweight:
            while self._estimate_memory_usage() > self.max_memory_bytes and len(self._cache) > 1:
                oldest_key, oldest_value = self._cache.popitem(last=False)
                logger.info(f"LRU cache evicted by memory: {oldest_key}")
                self._cleanup_value(oldest_value)
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of cache contents (expensive operation)."""
        if self.lightweight:
            return 0  # Skip expensive calculation
            
        total_bytes = 0
        for value in self._cache.values():
            if hasattr(value, 't3') and hasattr(value.t3, 'speaker_emb'):
                if value.t3.speaker_emb is not None:
                    total_bytes += value.t3.speaker_emb.numel() * value.t3.speaker_emb.element_size()
            if hasattr(value, 'gen') and isinstance(value.gen, dict):
                for tensor_value in value.gen.values():
                    if isinstance(tensor_value, torch.Tensor):
                        total_bytes += tensor_value.numel() * tensor_value.element_size()
        return total_bytes
    
    def _cleanup_value(self, value: Any) -> None:
        """Clean up GPU memory for evicted cache values."""
        if hasattr(value, 't3') and hasattr(value.t3, 'speaker_emb'):
            if value.t3.speaker_emb is not None and value.t3.speaker_emb.is_cuda:
                del value.t3.speaker_emb
        if hasattr(value, 'gen') and isinstance(value.gen, dict):
            for key in list(value.gen.keys()):
                if isinstance(value.gen[key], torch.Tensor) and value.gen[key].is_cuda:
                    del value.gen[key]
        # Force garbage collection to free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            for value in self._cache.values():
                self._cleanup_value(value)
            self._cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)
    
    def keys(self):
        """Get cache keys."""
        with self._lock:
            return list(self._cache.keys())
    
    def memory_usage_mb(self) -> float:
        """Get estimated memory usage in MB."""
        with self._lock:
            return self._estimate_memory_usage() / (1024 * 1024)


class ConditionalsCacheManager:
    """Thread-safe cache manager for conditionals with memory and disk storage"""
    
    def __init__(self, max_memory_cache_size: int = 10, max_memory_mb: int = 512, lightweight: bool = False):
        # Use LRU cache for memory management
        self._memory_cache = LRUCache(max_size=max_memory_cache_size, max_memory_mb=max_memory_mb, lightweight=lightweight)
        self._cache_lock = threading.RLock()  # RLock allows re-entrant locking
        self._current_loaded_cache_key = None
        self._disk_save_queue = []  # Track pending disk saves
        self.lightweight = lightweight
        
        # Log cache configuration
        mode = "lightweight" if lightweight else "full"
        logger.info(f"Initialized {mode} cache with max_size={max_memory_cache_size}, max_memory={max_memory_mb}MB")
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        with self._cache_lock:
            return {
                "memory_cache_size": self._memory_cache.size(),
                "memory_usage_mb": self._memory_cache.memory_usage_mb(),
                "pending_disk_saves": len(self._disk_save_queue),
                "current_loaded_key": self._current_loaded_cache_key
            }
    
    def clear_memory_cache(self) -> None:
        """Clear all memory cache entries."""
        with self._cache_lock:
            self._memory_cache.clear()
            logger.info("Memory cache cleared")
        
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
                    # Use LRU cache instead of direct dictionary
                    self._memory_cache.put(cache_key, cond_cls.clone())
                    logger.info(f"Saved conditionals object to LRU memory cache: {cache_key}")
                    
                    # Log cache statistics (skip expensive stats in lightweight mode)
                    if not self.lightweight:
                        stats = self.get_cache_stats()
                        logger.debug(f"Cache stats - Size: {stats['memory_cache_size']}, Memory: {stats['memory_usage_mb']:.1f}MB")
                
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
        """Try to load conditionals from LRU memory cache"""
        with self._cache_lock:
            cached_conditionals = self._memory_cache.get(cache_key)
            if cached_conditionals is not None:
                # Since device typically doesn't change during runtime, skip expensive device checks
                # Only do basic dtype verification for T3 speaker embeddings
                if (hasattr(cached_conditionals, 't3') and 
                    hasattr(cached_conditionals.t3, 'speaker_emb') and 
                    cached_conditionals.t3.speaker_emb is not None and
                    cached_conditionals.t3.speaker_emb.dtype != dtype):
                    cached_conditionals.t3.speaker_emb = cached_conditionals.t3.speaker_emb.to(dtype=dtype)
                
                model.set_conditionals(cached_conditionals)
                logger.info(f"Loaded conditionals from LRU memory cache: {cache_key}")
                self._current_loaded_cache_key = cache_key
                return True
        return False
    
    def _move_conditionals_to_device_dtype(self, cond_cls, device, dtype):
        """Safely move conditionals to target device and dtype."""
        try:
            # Move the entire conditionals object to device
            cond_cls = cond_cls.to(device)
            
            # Ensure T3 conditionals have correct dtype
            if hasattr(cond_cls, 't3') and cond_cls.t3 is not None:
                if hasattr(cond_cls.t3, 'speaker_emb') and cond_cls.t3.speaker_emb is not None:
                    cond_cls.t3.speaker_emb = cond_cls.t3.speaker_emb.to(dtype=dtype)
                if hasattr(cond_cls.t3, 'cond_prompt_speech_tokens') and cond_cls.t3.cond_prompt_speech_tokens is not None:
                    cond_cls.t3.cond_prompt_speech_tokens = cond_cls.t3.cond_prompt_speech_tokens.to(device=device)
                if hasattr(cond_cls.t3, 'emotion_adv') and cond_cls.t3.emotion_adv is not None:
                    cond_cls.t3.emotion_adv = cond_cls.t3.emotion_adv.to(device=device)
            
            # Ensure S3Gen conditionals are on correct device
            if hasattr(cond_cls, 'gen') and isinstance(cond_cls.gen, dict):
                for key, tensor_value in cond_cls.gen.items():
                    if isinstance(tensor_value, torch.Tensor):
                        cond_cls.gen[key] = tensor_value.to(device=device)
            
            return cond_cls
            
        except Exception as e:
            logger.error(f"Failed to move conditionals to device {device}: {e}")
            return cond_cls
    
    
    def _try_load_from_disk(self, cache_key, model, device, dtype, enable_memory_cache, quiet=False):
        """Try to load conditionals from disk cache"""
        cache_dir = get_cache_dir()
        cache_file = cache_dir.joinpath(cache_key + ".pt")
        
        if not cache_file.exists():
            logger.info(f"No disk cache file found for key: {cache_key} in {cache_file}")
            return False
        
        with safe_globals([T3Cond]):
            # Use device-aware map_location instead of hardcoded cuda
            map_location = device if device.type != 'mps' else 'cpu'
            kwargs = torch.load(cache_file, map_location=map_location, weights_only=True)
            cond_cls = Conditionals(T3Cond(**kwargs['t3']), kwargs['gen'])
        
        if not quiet:
            logger.info(f"Loaded conditionals from disk cache: {cache_key}")

        # Ensure correct device and dtype for all tensors
        cond_cls = self._move_conditionals_to_device_dtype(cond_cls, device, dtype)

        if hasattr(cond_cls, 't3'):
            model.set_conditionals(cond_cls)

        if enable_memory_cache:
            with self._cache_lock:
                # Use LRU cache instead of direct dictionary access
                self._memory_cache.put(cache_key, cond_cls)
                if not quiet:
                    logger.info(f"Cached conditionals in LRU memory: {cache_key}")

        with self._cache_lock:
            self._current_loaded_cache_key = cache_key

        return True
    


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


def get_cache_stats():
    """Get cache statistics for monitoring and debugging."""
    return _cache_manager.get_cache_stats()


def clear_memory_cache():
    """Clear all memory cache entries to free up memory."""
    _cache_manager.clear_memory_cache()


def configure_cache(max_size: int = 10, max_memory_mb: int = 512, lightweight: bool = False):
    """
    Reconfigure the global cache manager with new limits.
    
    Args:
        max_size: Maximum number of items to keep in memory cache
        max_memory_mb: Maximum memory usage in MB for cache
        lightweight: Skip expensive memory estimation (better performance)
    """
    global _cache_manager
    _cache_manager = ConditionalsCacheManager(max_size, max_memory_mb, lightweight)
    mode = "lightweight" if lightweight else "full"
    logger.info(f"Cache reconfigured: {mode} mode, max_size={max_size}, max_memory={max_memory_mb}MB")


def log_cache_stats():
    """Log current cache statistics."""
    stats = get_cache_stats()
    logger.info(f"Cache Statistics: {stats}")
    

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


def get_device_aware_cache_key(audio_path, uuid, device, exaggeration: float = None):
    """Generate a device-aware cache key for better cache separation."""
    base_key = get_cache_key(audio_path, uuid, exaggeration)
    if base_key is None:
        return None
    
    # Add device suffix to prevent cross-device cache conflicts
    device_suffix = str(device).replace(':', '_')  # cuda:0 -> cuda_0
    return f"{base_key}_{device_suffix}"

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
