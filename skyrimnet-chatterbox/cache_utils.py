import threading
import torch
import datetime
import functools
import os
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import gc

import psutil
import torchaudio
from loguru import logger
from torch.serialization import safe_globals

try:
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    from chatterbox.tts import Conditionals
except ImportError:
    from .chatterbox.models.t3.modules.cond_enc import T3Cond
    from .chatterbox.tts import Conditionals

CACHE_DIR = Path("cache")
WAV_OUTPUT_DIR = CACHE_DIR.joinpath("audio_output")


class ConditionalsCacheManager:
    """Simplified in-memory cache manager for conditionals."""
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()  # Reentrant lock for nested operations
        
    def get(self, language: str, cache_key: str) -> Optional[Any]:
        """Get conditionals from cache."""
        with self._lock:
            return self._cache.get(language, {}).get(cache_key)
    
    def set(self, language: str, cache_key: str, conditionals: Any) -> None:
        """Store conditionals in cache."""
        with self._lock:
            if language not in self._cache:
                self._cache[language] = {}
            self._cache[language][cache_key] = conditionals
    
    def get_all_keys(self) -> List[Tuple[str, str]]:
        """Get all cached keys as (language, key) tuples."""
        with self._lock:
            keys = []
            for lang, lang_cache in self._cache.items():
                for key in lang_cache.keys():
                    keys.append((lang, key))
            return keys
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        with self._lock:
            total_entries = sum(len(lang_cache) for lang_cache in self._cache.values())
            return {
                "total_entries": total_entries,
                "languages": len(self._cache),
                "languages_list": list(self._cache.keys())
            }
    
    def clear_memory_cache(self) -> None:
        """Clear all memory cache entries."""
        with self._lock:
            self._cache.clear()
            logger.info("Memory cache cleared")




# Global cache manager instance
_cache_manager = ConditionalsCacheManager()


def _save_pt_to_disk(language: str, cache_key: str, cond_cls):
    """Non-blocking worker function to save conditionals to disk"""
    try:
        cache_dir = get_conditionals_dir(language)
        cache_file = cache_dir.joinpath(cache_key + ".pt")
        arg_dict = dict(
            t3=cond_cls.t3.__dict__,
            gen=cond_cls.gen
        )
        torch.save(arg_dict, cache_file)
        logger.info(f"Saved conditionals to disk cache: {cache_key}")
    except Exception as e:
        logger.error(f"Failed to save conditionals to disk cache: {e}")


def _move_conditionals_to_device_dtype(cond_cls, device, dtype):
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


def save_conditionals_cache(language: str, cache_key: str, cond_cls, enable_memory_cache=True, enable_disk_cache=True):
    """Save prepared conditionals to disk (non-blocking) and memory"""
    if cache_key is None:
        return False
    
    try:
        if enable_memory_cache:
            _cache_manager.set(language, cache_key, cond_cls.clone())
            logger.info(f"Saved conditionals to memory cache: {language}/{cache_key}")
        
        # Queue disk save (non-blocking)
        if enable_disk_cache:
            threading.Thread(
                target=_save_pt_to_disk,
                args=(language, cache_key, cond_cls),
                daemon=True,
                name=f"DiskCacheSave-{cache_key}"
            ).start()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save conditionals cache: {e}")
        return False


def load_conditionals_cache(language: str, cache_key: str, model, device, dtype, enable_memory_cache=True, enable_disk_cache=True):
    """Load prepared conditionals from memory or disk cache"""
    if cache_key is None:
        logger.info("No cache key provided, cannot load conditionals")
        return False
    
    try:
        # Try memory cache first
        if enable_memory_cache:
            cached = _cache_manager.get(language, cache_key)
            if cached is not None:
                # Ensure correct dtype for T3 speaker embeddings
                if (hasattr(cached, 't3') and 
                    hasattr(cached.t3, 'speaker_emb') and 
                    cached.t3.speaker_emb is not None and
                    cached.t3.speaker_emb.dtype != dtype):
                    cached.t3.speaker_emb = cached.t3.speaker_emb.to(dtype=dtype)
                
                model.set_conditionals(cached)
                logger.info(f"Loaded conditionals from memory cache: {language}/{cache_key}")
                return True
        
        # Try disk cache
        if enable_disk_cache:
            cache_dir = get_conditionals_dir(language)
            cache_file = cache_dir.joinpath(cache_key + ".pt")
            
            if cache_file.exists():
                with safe_globals([T3Cond]):
                    # Use device-aware map_location
                    device_obj = torch.device(device) if isinstance(device, str) else device
                    map_location = device if device_obj.type != 'mps' else 'cpu'
                    kwargs = torch.load(cache_file, map_location=map_location, weights_only=True)
                    cond_cls = Conditionals(T3Cond(**kwargs['t3']), kwargs['gen'])
                
                logger.info(f"Loaded conditionals from disk cache: {language}/{cache_key}")
                
                # Ensure correct device and dtype for all tensors
                cond_cls = _move_conditionals_to_device_dtype(cond_cls, device, dtype)
                
                if hasattr(cond_cls, 't3'):
                    model.set_conditionals(cond_cls)
                
                # Cache in memory for future use
                if enable_memory_cache:
                    _cache_manager.set(language, cache_key, cond_cls)
                
                return True
        
        logger.info(f"No cached conditionals found for key: {language}/{cache_key}")
        return False
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        logger.error(f"Failed to load conditionals cache: {e}")
        return False


def get_cache_stats():
    """Get cache statistics for monitoring and debugging."""
    return _cache_manager.get_cache_stats()


def clear_memory_cache():
    """Clear all memory cache entries to free up memory."""
    _cache_manager.clear_memory_cache()


def get_all_cache_keys() -> List[Tuple[str, str]]:
    """Return a list of all cached conditional keys as (language, key) tuples."""
    return _cache_manager.get_all_keys()


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
def get_conditionals_dir(language: str = "en"):
    """Get or create the conditionals cache directory for a specific language"""
    cache_dir = CACHE_DIR.joinpath("conditionals").joinpath(language)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

@functools.cache
def get_cache_key(audio_path, exaggeration: float = None):
    """Generate a cache key based on audio file, and exaggeration"""
    if audio_path is None:
        return None

    cache_prefix = Path(audio_path).stem

    if exaggeration is None:
        cache_key = f"{cache_prefix}"
    else:
        cache_key = f"{cache_prefix}_{exaggeration:.2f}"

    return cache_key


def get_device_aware_cache_key(audio_path, device, exaggeration: float = None):
    """Generate a device-aware cache key for better cache separation."""
    base_key = get_cache_key(audio_path, exaggeration)
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

def save_torchaudio_wav(wav_tensor, sr, audio_path):
    """Save a tensor as a WAV file using torchaudio"""

    formatted_now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{formatted_now_time}_{get_cache_key(audio_path)}"
    path = get_wavout_dir() / f"{filename}.wav"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torchaudio.save(path, wav_tensor.cpu(), sr, encoding="PCM_S")
    return path.resolve()

def init_conditional_memory_cache(model, device, dtype, supported_languages: List[str] = ["en"]) -> None:
    """Initialize conditional cache from disk for all supported languages."""
    for language in supported_languages:
        cache_dir = get_conditionals_dir(language)
        for filename in cache_dir.glob("*.pt"):
            cache_key = filename.stem
            try:
                # Load directly into memory cache without setting on model
                with safe_globals([T3Cond]):
                    device_obj = torch.device(device) if isinstance(device, str) else device
                    map_location = device if device_obj.type != 'mps' else 'cpu'
                    kwargs = torch.load(filename, map_location=map_location, weights_only=True)
                    cond_cls = Conditionals(T3Cond(**kwargs['t3']), kwargs['gen'])
                
                # Ensure correct device and dtype
                cond_cls = _move_conditionals_to_device_dtype(cond_cls, device, dtype)
                
                # Store in memory cache
                _cache_manager.set(language, cache_key, cond_cls)
                
            except Exception as e:
                logger.error(f"Failed to load conditionals from {filename}: {e}")
    
    stats = get_cache_stats()
    logger.info(
        f"Loaded conditionals from disk with {stats['total_entries']} entries across languages: {stats['languages_list']}")

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

def clear_cache_files(language: Optional[str] = None):
    """Remove all .pt files in the cache directory
    
    Args:
        language: If specified, only clear cache for that language. Otherwise clear all languages.
    """
    cache_base = Path("cache").joinpath("conditionals")
    if not cache_base.exists():
        logger.info(f"Cache directory {cache_base} does not exist, nothing to clear")
        return 0
    
    removed_count = 0
    try:
        if language is not None:
            # Clear specific language
            cache_dir = get_conditionals_dir(language)
            for pt_file in cache_dir.glob("*.pt"):
                pt_file.unlink()
                logger.info(f"Removed cache file: {pt_file}")
                removed_count += 1
        else:
            # Clear all languages
            for lang_dir in cache_base.iterdir():
                if lang_dir.is_dir():
                    for pt_file in lang_dir.glob("*.pt"):
                        pt_file.unlink()
                        logger.info(f"Removed cache file: {pt_file}")
                        removed_count += 1
        
        # Clear memory cache as well since disk files are gone
        clear_memory_cache()
        
        logger.info(f"Cleared {removed_count} cache files")
        return removed_count
        
    except Exception as e:
        logger.error(f"Failed to clear cache files: {e}")
        return 0
