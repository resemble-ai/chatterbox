"""chatterbox/__init__.py resolves __version__ from the installed package
metadata. If chatterbox-tts isn't registered as an installed distribution
(e.g. the package directory is used directly, without `pip install -e .`),
importlib.metadata.version() raises PackageNotFoundError and `import
chatterbox` used to crash before reaching ChatterboxTTS/ChatterboxVC.

This test exercises just the version-resolution block from __init__.py in
isolation, so it doesn't need chatterbox's heavy ML dependencies installed.
"""

import importlib.metadata
import unittest
from pathlib import Path
from unittest import mock

_INIT_PY = Path(__file__).resolve().parents[1] / "src" / "chatterbox" / "__init__.py"


def _version_block_source() -> str:
    lines = _INIT_PY.read_text().splitlines()
    end = next(i for i, line in enumerate(lines) if line.startswith("from .tts"))
    return "\n".join(lines[:end])


class VersionFallbackTests(unittest.TestCase):
    def test_falls_back_when_package_metadata_is_missing(self) -> None:
        with mock.patch.object(
            importlib.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError
        ):
            namespace: dict = {}
            exec(_version_block_source(), namespace)  # noqa: S102
        self.assertEqual(namespace["__version__"], "0.1.7")

    def test_uses_installed_metadata_when_present(self) -> None:
        with mock.patch.object(importlib.metadata, "version", return_value="9.9.9"):
            namespace: dict = {}
            exec(_version_block_source(), namespace)  # noqa: S102
        self.assertEqual(namespace["__version__"], "9.9.9")


if __name__ == "__main__":
    unittest.main()
