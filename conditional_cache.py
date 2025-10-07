"""
Production-ready conditional caching utility for Chatterbox TTS.

This module provides efficient caching of audio conditionals to avoid
redundant processing when generating multiple texts with the same voice.
"""

from pathlib import Path
from typing import Optional, Dict, Tuple
import hashlib
from chatterbox.mtl_tts import Conditionals, ChatterboxMultilingualTTS


class ConditionalCache:
    """
    Cache manager for TTS conditionals.
    
    Stores conditionals in memory and optionally persists them to disk.
    Conditionals are keyed by (audio_path, exaggeration) tuples.
    
    Example:
        >>> cache = ConditionalCache(cache_dir="./cache")
        >>> model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
        >>> 
        >>> # First call prepares conditionals
        >>> cache.get_or_prepare(model, "voice.wav", exaggeration=0.5)
        >>> wav1 = model.generate("Text 1", language_id="en")
        >>> 
        >>> # Subsequent calls reuse cached conditionals
        >>> cache.get_or_prepare(model, "voice.wav", exaggeration=0.5)
        >>> wav2 = model.generate("Text 2", language_id="en")  # Much faster!
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, auto_save: bool = True):
        """
        Initialize the conditional cache.
        
        Args:
            cache_dir: Directory to store cached conditionals. If None, only uses memory cache.
            auto_save: If True, automatically save new conditionals to disk.
        """
        self.memory_cache: Dict[Tuple[str, float], Conditionals] = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.auto_save = auto_save
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _make_cache_key(self, audio_path: str, exaggeration: float) -> Tuple[str, float]:
        """Create a cache key from audio path and exaggeration."""
        # Normalize path to absolute path for consistent caching
        audio_path = str(Path(audio_path).resolve())
        return (audio_path, round(exaggeration, 4))  # Round to avoid float precision issues
    
    def _get_cache_filename(self, audio_path: str, exaggeration: float) -> Path:
        """Generate a filename for disk cache based on audio path and exaggeration."""
        # Create hash of audio path to avoid filesystem issues with long paths
        path_hash = hashlib.md5(audio_path.encode()).hexdigest()[:16]
        audio_name = Path(audio_path).stem
        exag_str = f"{exaggeration:.4f}".replace(".", "_")
        return self.cache_dir / f"cond_{audio_name}_{path_hash}_exag{exag_str}.pt"
    
    def get_or_prepare(
        self,
        model: ChatterboxMultilingualTTS,
        audio_path: str,
        exaggeration: float = 0.5,
        force_refresh: bool = False,
        verbose: bool = True
    ) -> Conditionals:
        """
        Get conditionals from cache or prepare them if not cached.
        
        Args:
            model: The TTS model to use for preparation.
            audio_path: Path to the reference audio file.
            exaggeration: Exaggeration parameter.
            force_refresh: If True, ignore cache and re-prepare conditionals.
            verbose: If True, print cache status messages.
        
        Returns:
            Conditionals object ready to use.
        """
        cache_key = self._make_cache_key(audio_path, exaggeration)
        
        # Check memory cache first
        if not force_refresh and cache_key in self.memory_cache:
            if verbose:
                print(f"✓ Memory cache hit: {Path(audio_path).name} (exag={exaggeration})")
            model.conds = self.memory_cache[cache_key]
            return model.conds
        
        # Check disk cache
        if not force_refresh and self.cache_dir:
            cache_file = self._get_cache_filename(audio_path, exaggeration)
            if cache_file.exists():
                if verbose:
                    print(f"✓ Disk cache hit: {cache_file.name}")
                try:
                    conds = Conditionals.load(cache_file, map_location=str(model.device))
                    conds = conds.to(model.device)
                    self.memory_cache[cache_key] = conds
                    model.conds = conds
                    return model.conds
                except Exception as e:
                    if verbose:
                        print(f"⚠ Failed to load from disk cache: {e}")
        
        # Cache miss - prepare new conditionals
        if verbose:
            print(f"⊙ Cache miss - preparing conditionals: {Path(audio_path).name} (exag={exaggeration})")
        
        model.prepare_conditionals(audio_path, exaggeration=exaggeration)
        
        # Store in memory cache
        self.memory_cache[cache_key] = model.conds
        
        # Optionally save to disk
        if self.auto_save and self.cache_dir:
            cache_file = self._get_cache_filename(audio_path, exaggeration)
            try:
                model.conds.save(cache_file)
                if verbose:
                    print(f"✓ Saved to disk cache: {cache_file.name}")
            except Exception as e:
                if verbose:
                    print(f"⚠ Failed to save to disk cache: {e}")
        
        return model.conds
    
    def save(self, audio_path: str, exaggeration: float, filepath: Path):
        """
        Manually save cached conditionals to a specific file.
        
        Args:
            audio_path: The audio path used to cache the conditionals.
            exaggeration: The exaggeration value used.
            filepath: Path where to save the conditionals.
        """
        cache_key = self._make_cache_key(audio_path, exaggeration)
        if cache_key in self.memory_cache:
            self.memory_cache[cache_key].save(filepath)
        else:
            raise ValueError(f"No cached conditionals found for {audio_path} with exaggeration={exaggeration}")
    
    def load(self, filepath: Path, audio_path: str, exaggeration: float, device: str = "cpu"):
        """
        Manually load conditionals from a file into the cache.
        
        Args:
            filepath: Path to the saved conditionals file.
            audio_path: The audio path to associate with these conditionals.
            exaggeration: The exaggeration value to associate with these conditionals.
            device: Device to load the conditionals to.
        """
        cache_key = self._make_cache_key(audio_path, exaggeration)
        conds = Conditionals.load(filepath, map_location=device)
        self.memory_cache[cache_key] = conds
        return conds
    
    def clear_memory(self):
        """Clear the in-memory cache."""
        self.memory_cache.clear()
    
    def clear_disk(self):
        """Delete all cached files from disk."""
        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("cond_*.pt"):
                cache_file.unlink()
    
    def clear_all(self):
        """Clear both memory and disk cache."""
        self.clear_memory()
        self.clear_disk()
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about the cache."""
        stats = {
            "memory_cached_items": len(self.memory_cache),
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
        }
        
        if self.cache_dir and self.cache_dir.exists():
            cache_files = list(self.cache_dir.glob("cond_*.pt"))
            stats["disk_cached_items"] = len(cache_files)
            stats["disk_cache_size_mb"] = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
        else:
            stats["disk_cached_items"] = 0
            stats["disk_cache_size_mb"] = 0.0
        
        return stats
    
    def __repr__(self):
        stats = self.get_cache_stats()
        return (f"ConditionalCache("
                f"memory={stats['memory_cached_items']}, "
                f"disk={stats['disk_cached_items']}, "
                f"size={stats['disk_cache_size_mb']:.2f}MB)")


# Example usage
if __name__ == "__main__":
    import torch
    import torchaudio as ta
    
    # Setup
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cache = ConditionalCache(cache_dir="./tts_cache", auto_save=True)
    
    print("Loading model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    
    # Define voice and texts
    voice_file = "julia-whelan.wav"
    texts = [
        "Hello, this is a test.",
        "Now generating another sentence.",
        "And one more for good measure.",
    ]
    
    print(f"\n{cache}\n")
    
    # Generate multiple texts with caching
    for i, text in enumerate(texts, 1):
        print(f"\n[{i}/{len(texts)}] '{text}'")
        
        # This will prepare conditionals on first call, then reuse cache
        cache.get_or_prepare(model, voice_file, exaggeration=0.5)
        
        wav = model.generate(text, language_id="en", cfg_weight=0.5)
        ta.save(f"output_{i}.wav", wav, model.sr)
    
    print(f"\n{cache}")
    print("\nCache statistics:")
    for key, value in cache.get_cache_stats().items():
        print(f"  {key}: {value}")
