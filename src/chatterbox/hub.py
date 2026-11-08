"""Shared, revision-consistent checkpoint downloads."""

import logging

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


def download_snapshot(repo_id, allow_patterns, *, revision=None, cache_dir=None,
                      local_files_only=False, token=None):
    """Fetch one snapshot; accept private mirrors and Hugging Face's offline cache."""
    kwargs = dict(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        allow_patterns=allow_patterns,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    try:
        return snapshot_download(**kwargs)
    except Exception as exc:
        if "xet" not in str(exc).lower() and "hex hash" not in str(exc).lower():
            raise
        # Retain upstream's HTTP fallback for broken Xet installations, but restore
        # the process setting afterward so unrelated downloads are unaffected.
        import huggingface_hub.constants as constants

        previous = constants.HF_HUB_DISABLE_XET
        logger.warning("Xet download failed; retrying over HTTP: %s", exc)
        constants.HF_HUB_DISABLE_XET = True
        try:
            return snapshot_download(**kwargs)
        finally:
            constants.HF_HUB_DISABLE_XET = previous
