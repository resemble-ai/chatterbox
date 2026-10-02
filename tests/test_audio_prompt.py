import functools
import http.server
import importlib.util
import threading
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "audio_prompt", Path(__file__).resolve().parents[1] / "src" / "chatterbox" / "audio_prompt.py"
)
audio_prompt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audio_prompt)


@pytest.fixture
def server(tmp_path):
    served = tmp_path / "served"
    served.mkdir()
    (served / "voice.flac").write_bytes(b"fLaC-fake-audio")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(served))
    handler.log_message = lambda *a, **k: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_local_path_and_empty_values_pass_through(tmp_path):
    assert audio_prompt.resolve_audio_prompt_path("/some/voice.wav", tmp_path) == "/some/voice.wav"
    assert audio_prompt.resolve_audio_prompt_path(None, tmp_path) is None
    assert audio_prompt.resolve_audio_prompt_path("", tmp_path) == ""


def test_url_is_downloaded_to_a_local_file_and_cached(server, tmp_path):
    cache = tmp_path / "cache"
    url = f"{server}/voice.flac"
    first = audio_prompt.resolve_audio_prompt_path(url, cache)
    assert not audio_prompt.is_url(first)
    assert first.endswith(".flac")
    assert Path(first).read_bytes() == b"fLaC-fake-audio"
    mtime = Path(first).stat().st_mtime_ns
    assert audio_prompt.resolve_audio_prompt_path(url, cache) == first
    assert Path(first).stat().st_mtime_ns == mtime


def test_failed_download_leaves_no_partial_file(server, tmp_path):
    cache = tmp_path / "cache"
    with pytest.raises(Exception):
        audio_prompt.resolve_audio_prompt_path(f"{server}/missing.flac", cache)
    assert list(cache.iterdir()) == []
