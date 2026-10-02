"""Helpers for reference-voice prompts that may be given as a local path or an http(s) URL."""
import hashlib
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

_CACHE_DIR = Path(tempfile.gettempdir()) / "chatterbox_audio_prompts"


def is_url(value: str) -> bool:
    return urllib.parse.urlparse(str(value)).scheme in ("http", "https")


def resolve_audio_prompt_path(
    path_or_url: Optional[str], cache_dir: Optional[os.PathLike] = None
) -> Optional[str]:
    """Return a local file path for a reference voice.

    Local paths (and empty values) are returned unchanged. http(s) URLs are downloaded once
    into a cache directory and the cached file's path is returned, because `librosa.load`
    can only open local files.
    """
    if not path_or_url or not str(path_or_url).strip():
        return path_or_url
    value = str(path_or_url).strip()
    if not is_url(value):
        return path_or_url

    cache = Path(cache_dir) if cache_dir else _CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    suffix = Path(urllib.parse.urlparse(value).path).suffix or ".wav"
    target = cache / (hashlib.sha256(value.encode()).hexdigest()[:24] + suffix)
    if target.exists() and target.stat().st_size > 0:
        return str(target)

    tmp = target.with_name(target.name + f".{os.getpid()}.part")
    try:
        with urllib.request.urlopen(value, timeout=30) as response, open(tmp, "wb") as out:
            out.write(response.read())
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
    return str(target)
