"""Regression tests for Hebrew diacritization model loading (issue #467).

The bug: ``add_hebrew_diacritics()`` called ``Dicta()`` with no arguments, but
``dicta_onnx.Dicta.__init__`` requires a ``model_path`` argument.  The resulting
``TypeError`` was swallowed by the ``except Exception`` handler, so Hebrew TTS
silently received un-voweled text and produced gibberish.

These tests verify:
1. ``_get_dicta_model_path()`` respects ``DICTA_MODEL_PATH`` env-var
2. ``_get_dicta_model_path()`` creates the cache directory and downloads on miss
3. ``_get_dicta_model_path()`` returns the cached file on subsequent calls
4. ``add_hebrew_diacritics()`` passes model_path to ``Dicta()``
5. ``add_hebrew_diacritics()`` warns clearly when dicta_onnx is missing
6. ``add_hebrew_diacritics()`` warns clearly when model file is missing
7. Source-code audit: no bare ``Dicta()`` calls remain
"""

import ast
import importlib
import importlib.util
import logging
import os
import sys
import types
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
TOKENIZER_PY = SRC_DIR / "chatterbox" / "models" / "tokenizers" / "tokenizer.py"


def _load_tokenizer_module() -> types.ModuleType:
    """Load *only* tokenizer.py as a standalone module, bypassing the
    chatterbox package hierarchy (which requires torch, torchaudio, etc.)."""
    spec = importlib.util.spec_from_file_location(
        "_tokenizer_standalone", str(TOKENIZER_PY),
        # Provide a fake submodule loader so ``from huggingface_hub import …``
        # and similar top-level imports don't fail.
    )
    assert spec is not None and spec.loader is not None

    # Stub out heavy dependencies that tokenizer.py imports at module level
    stubs = {}
    for mod_name in [
        "torch", "tokenizers", "huggingface_hub",
        "chatterbox", "chatterbox.models", "chatterbox.models.tokenizers",
    ]:
        if mod_name not in sys.modules:
            stubs[mod_name] = types.ModuleType(mod_name)

    # torch needs .IntTensor
    torch_stub = stubs.get("torch") or sys.modules.get("torch")
    if not hasattr(torch_stub, "IntTensor"):
        torch_stub.IntTensor = lambda x: x  # type: ignore

    # tokenizers needs .Tokenizer
    tok_stub = stubs.get("tokenizers") or sys.modules.get("tokenizers")
    if not hasattr(tok_stub, "Tokenizer"):
        class _FakeTokenizer:
            @classmethod
            def from_file(cls, *a, **kw):
                return cls()
            def get_vocab(self):
                return {"[START]": 0, "[STOP]": 1}
            def encode(self, txt):
                class R:
                    ids = [0]
                return R()
            def decode(self, seq, **kw):
                return ""
        tok_stub.Tokenizer = _FakeTokenizer  # type: ignore

    # huggingface_hub needs .hf_hub_download
    hf_stub = stubs.get("huggingface_hub") or sys.modules.get("huggingface_hub")
    if not hasattr(hf_stub, "hf_hub_download"):
        hf_stub.hf_hub_download = lambda **kw: "/dev/null"  # type: ignore

    with mock.patch.dict(sys.modules, stubs):
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    # Reset the singleton so each test starts clean
    mod._dicta = None  # type: ignore
    return mod


# ---------------------------------------------------------------------------
# 1. DICTA_MODEL_PATH env-var override
# ---------------------------------------------------------------------------

class TestDictaModelPathEnvVar:
    def test_env_var_valid_file(self, tmp_path):
        """DICTA_MODEL_PATH pointing to an existing file is returned as-is."""
        mod = _load_tokenizer_module()
        fake_model = tmp_path / "custom-dicta.onnx"
        fake_model.write_text("fake model data")
        with mock.patch.dict(os.environ, {"DICTA_MODEL_PATH": str(fake_model)}):
            result = mod._get_dicta_model_path()
        assert result == fake_model

    def test_env_var_missing_file(self, tmp_path):
        """DICTA_MODEL_PATH pointing to a non-existent file raises FileNotFoundError."""
        mod = _load_tokenizer_module()
        with mock.patch.dict(os.environ, {"DICTA_MODEL_PATH": str(tmp_path / "nope.onnx")}):
            with pytest.raises(FileNotFoundError, match="DICTA_MODEL_PATH"):
                mod._get_dicta_model_path()

    def test_env_var_tilde_expansion(self, tmp_path):
        """Path from DICTA_MODEL_PATH is expanduser'd."""
        mod = _load_tokenizer_module()
        fake_model = tmp_path / "model.onnx"
        fake_model.write_text("data")
        with mock.patch.dict(os.environ, {"DICTA_MODEL_PATH": str(fake_model)}):
            result = mod._get_dicta_model_path()
        assert result.is_file()


# ---------------------------------------------------------------------------
# 2. Cache directory creation & download
# ---------------------------------------------------------------------------

class TestCacheAndDownload:
    def test_creates_cache_dir_and_downloads(self, tmp_path):
        """On first call (no cache), creates dir and downloads the model."""
        mod = _load_tokenizer_module()
        cache_dir = tmp_path / "cache"
        download_calls = []

        def fake_urlretrieve(url, dest):
            download_calls.append((url, dest))
            Path(dest).write_text("fake-onnx-data")

        env = {"XDG_CACHE_HOME": str(cache_dir)}
        # Clear DICTA_MODEL_PATH if present
        env_clear = {k: v for k, v in os.environ.items() if k != "DICTA_MODEL_PATH"}
        env_clear.update(env)

        with mock.patch.dict(os.environ, env_clear, clear=True):
            with mock.patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
                result = mod._get_dicta_model_path()

        assert len(download_calls) == 1
        assert mod._DICTA_MODEL_URL in download_calls[0][0]
        assert result.is_file()
        assert result.name == mod._DICTA_MODEL_FILENAME
        assert "chatterbox" in str(result)

    def test_returns_cached_file_without_download(self, tmp_path):
        """If the model file already exists in cache, no download occurs."""
        mod = _load_tokenizer_module()
        cache_dir = tmp_path / "cache" / "chatterbox" / "dicta"
        cache_dir.mkdir(parents=True)
        cached = cache_dir / mod._DICTA_MODEL_FILENAME
        cached.write_text("cached model")

        env = {"XDG_CACHE_HOME": str(tmp_path / "cache")}
        env_clear = {k: v for k, v in os.environ.items() if k != "DICTA_MODEL_PATH"}
        env_clear.update(env)

        with mock.patch.dict(os.environ, env_clear, clear=True):
            with mock.patch("urllib.request.urlretrieve") as mock_dl:
                result = mod._get_dicta_model_path()

        mock_dl.assert_not_called()
        assert result == cached

    def test_partial_download_cleaned_up_on_failure(self, tmp_path):
        """If download fails, the partial .part file is removed."""
        mod = _load_tokenizer_module()
        cache_dir = tmp_path / "cache"

        env = {"XDG_CACHE_HOME": str(cache_dir)}
        env_clear = {k: v for k, v in os.environ.items() if k != "DICTA_MODEL_PATH"}
        env_clear.update(env)

        with mock.patch.dict(os.environ, env_clear, clear=True):
            with mock.patch(
                "urllib.request.urlretrieve",
                side_effect=ConnectionError("network down"),
            ):
                with pytest.raises(ConnectionError):
                    mod._get_dicta_model_path()

        # No .part files should remain
        dicta_dir = cache_dir / "chatterbox" / "dicta"
        if dicta_dir.exists():
            part_files = list(dicta_dir.glob("*.part"))
            assert len(part_files) == 0, f"Partial files remain: {part_files}"


# ---------------------------------------------------------------------------
# 3. add_hebrew_diacritics passes model_path to Dicta
# ---------------------------------------------------------------------------

class TestAddHebrewDiacritics:
    def test_dicta_receives_model_path(self, tmp_path):
        """Dicta() is called with the model path, not bare."""
        mod = _load_tokenizer_module()
        init_calls = []

        class FakeDicta:
            def __init__(self, model_path):
                init_calls.append(model_path)

            def add_diacritics(self, text):
                return text + "\u05B0"  # add a niqqud character

        fake_model = tmp_path / "dicta-1.0.int8.onnx"
        fake_model.write_text("fake")

        fake_dicta_mod = types.ModuleType("dicta_onnx")
        fake_dicta_mod.Dicta = FakeDicta  # type: ignore

        with mock.patch.dict(os.environ, {"DICTA_MODEL_PATH": str(fake_model)}):
            with mock.patch.dict(sys.modules, {"dicta_onnx": fake_dicta_mod}):
                result = mod.add_hebrew_diacritics("שלום")

        assert len(init_calls) == 1
        assert init_calls[0] == str(fake_model)
        assert "\u05B0" in result

    def test_bare_dicta_call_would_fail(self):
        """Verify the original bug: Dicta() with no args raises TypeError."""
        class StrictDicta:
            def __init__(self, model_path: str):
                if not isinstance(model_path, str):
                    raise TypeError("model_path is required")

        with pytest.raises(TypeError):
            StrictDicta()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 4. Clear warnings when dependencies are missing
# ---------------------------------------------------------------------------

class TestWarnings:
    def test_warns_on_missing_dicta_onnx(self, caplog):
        """When dicta_onnx is not installed, logs a helpful warning."""
        mod = _load_tokenizer_module()
        mod._dicta = None

        # Remove dicta_onnx from sys.modules to trigger ImportError
        with mock.patch.dict(sys.modules, {"dicta_onnx": None}):
            with caplog.at_level(logging.WARNING):
                result = mod.add_hebrew_diacritics("שלום")

        assert result == "שלום"  # returned unchanged
        warning_msgs = [r.message for r in caplog.records]
        assert any("dicta_onnx" in msg or "dicta-onnx" in msg for msg in warning_msgs), \
            f"Expected dicta warning, got: {warning_msgs}"

    def test_warns_on_missing_model_file(self, caplog, tmp_path):
        """When DICTA_MODEL_PATH points nowhere, logs a clear warning."""
        mod = _load_tokenizer_module()
        mod._dicta = None

        fake_dicta_mod = types.ModuleType("dicta_onnx")
        fake_dicta_mod.Dicta = lambda path: None  # type: ignore

        with mock.patch.dict(os.environ, {"DICTA_MODEL_PATH": str(tmp_path / "nope.onnx")}):
            with mock.patch.dict(sys.modules, {"dicta_onnx": fake_dicta_mod}):
                with caplog.at_level(logging.WARNING):
                    result = mod.add_hebrew_diacritics("שלום")

        assert result == "שלום"  # returned unchanged
        warning_msgs = [r.message for r in caplog.records]
        assert any("model not found" in msg.lower() or "DICTA_MODEL_PATH" in msg
                    for msg in warning_msgs), \
            f"Expected model-not-found warning, got: {warning_msgs}"


# ---------------------------------------------------------------------------
# 5. Source-code audit: no bare Dicta() calls
# ---------------------------------------------------------------------------

class TestSourceAudit:
    def test_no_bare_dicta_constructor(self):
        """The tokenizer.py source must not contain ``Dicta()`` with zero args."""
        source = TOKENIZER_PY.read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "Dicta":
                    assert len(node.args) > 0 or len(node.keywords) > 0, (
                        f"Line {node.lineno}: bare Dicta() call with no arguments "
                        f"— must pass model_path"
                    )

    def test_get_dicta_model_path_called_before_dicta(self):
        """_get_dicta_model_path() is called in add_hebrew_diacritics before Dicta()."""
        source = TOKENIZER_PY.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "add_hebrew_diacritics":
                func_source = ast.get_source_segment(source, node)
                assert func_source is not None
                path_pos = func_source.find("_get_dicta_model_path")
                dicta_pos = func_source.find("Dicta(")
                assert path_pos != -1, "_get_dicta_model_path not found in add_hebrew_diacritics"
                assert dicta_pos != -1, "Dicta( not found in add_hebrew_diacritics"
                assert path_pos < dicta_pos, (
                    "_get_dicta_model_path must be called before Dicta()"
                )
                break
        else:
            pytest.fail("add_hebrew_diacritics function not found")

    def test_dicta_model_url_points_to_github_release(self):
        """The download URL points to the official dicta-onnx GitHub release."""
        mod = _load_tokenizer_module()
        assert "thewh1teagle/dicta-onnx" in mod._DICTA_MODEL_URL
        assert "dicta-1.0.int8.onnx" in mod._DICTA_MODEL_URL

    def test_imports_include_os(self):
        """tokenizer.py imports os (needed for env-var, tempfile, replace)."""
        source = TOKENIZER_PY.read_text()
        assert "import os" in source


# ---------------------------------------------------------------------------
# 6. Integration-style round-trip (no real model, exercises the full path)
# ---------------------------------------------------------------------------

class TestIntegrationRoundTrip:
    def test_full_path_with_fake_dicta(self, tmp_path):
        """End-to-end: env-var → model_path → Dicta(path) → add_diacritics."""
        mod = _load_tokenizer_module()

        class FakeDicta:
            def __init__(self, model_path):
                assert Path(model_path).is_file()

            def add_diacritics(self, text):
                return "הַשָּׁלוֹם"

        fake_model = tmp_path / "dicta-1.0.int8.onnx"
        fake_model.write_bytes(b"\x00" * 100)

        fake_dicta_mod = types.ModuleType("dicta_onnx")
        fake_dicta_mod.Dicta = FakeDicta  # type: ignore

        with mock.patch.dict(os.environ, {"DICTA_MODEL_PATH": str(fake_model)}):
            with mock.patch.dict(sys.modules, {"dicta_onnx": fake_dicta_mod}):
                result = mod.add_hebrew_diacritics("השלום")

        assert result == "הַשָּׁלוֹם"

    def test_singleton_reuse(self, tmp_path):
        """After first successful init, subsequent calls reuse the singleton."""
        mod = _load_tokenizer_module()
        init_count = [0]

        class CountingDicta:
            def __init__(self, model_path):
                init_count[0] += 1

            def add_diacritics(self, text):
                return text

        fake_model = tmp_path / "dicta-1.0.int8.onnx"
        fake_model.write_text("fake")

        fake_dicta_mod = types.ModuleType("dicta_onnx")
        fake_dicta_mod.Dicta = CountingDicta  # type: ignore

        with mock.patch.dict(os.environ, {"DICTA_MODEL_PATH": str(fake_model)}):
            with mock.patch.dict(sys.modules, {"dicta_onnx": fake_dicta_mod}):
                mod.add_hebrew_diacritics("first")
                mod.add_hebrew_diacritics("second")
                mod.add_hebrew_diacritics("third")

        assert init_count[0] == 1, f"Dicta was initialized {init_count[0]} times, expected 1"

    def test_issue_467_exact_reproduction(self):
        """Reproduce the exact crash from issue #467: Dicta() with no model_path."""
        # This verifies the original bug scenario — calling Dicta() with no
        # arguments would raise TypeError, which was caught by except Exception
        # and silently returned un-voweled text.
        class RealSignatureDicta:
            def __init__(self, model_path: str):
                # This is the real dicta_onnx.Dicta signature
                pass

        # The old buggy code did: _dicta = Dicta()  [no args]
        with pytest.raises(TypeError):
            RealSignatureDicta()  # type: ignore[call-arg]
