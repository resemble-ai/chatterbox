"""Helpers for locating and loading checkpoint files.

Kept free of model imports so the loading rules can be exercised without
instantiating (or downloading) any model.
"""
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import torch
from safetensors.torch import load_file as load_safetensors

PathLike = Union[str, Path]

STATE_DICT_SUFFIXES = (".pt", ".safetensors")


def resolve_checkpoint_name(
    name: Optional[str],
    aliases: Mapping[str, str],
    default: str,
    what: str = "checkpoint",
) -> str:
    """Turn a user-facing checkpoint selector into a filename.

    ``name`` may be ``None`` (use ``default``), an alias key of ``aliases`` (for
    example ``"v3"``), or an explicit ``.pt`` / ``.safetensors`` filename.
    """
    if name is None:
        return default
    if name in aliases:
        return aliases[name]
    if name.endswith(STATE_DICT_SUFFIXES):
        return name
    raise ValueError(
        f"Unknown {what} '{name}'. "
        f"Expected one of {sorted(aliases)} or a .pt / .safetensors filename."
    )


def load_state_dict_file(path: PathLike, map_location=None) -> dict:
    """Load a ``.safetensors`` or torch ``.pt`` state dict from ``path``."""
    path = Path(path)
    if path.suffix == ".safetensors":
        return load_safetensors(str(path))
    return torch.load(path, map_location=map_location, weights_only=True)


def load_state_dict_checked(
    module: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    ignore_missing: Iterable[str] = (),
    what: str = "checkpoint",
) -> torch.nn.Module:
    """Load ``state_dict`` into ``module``, tolerating only ``ignore_missing``.

    A plain ``strict=False`` load would also hide genuinely missing or
    unexpected weights, so this re-checks the result and raises for anything
    other than the keys the module declares as derived buffers.
    """
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    ignore = set(ignore_missing)
    missing = [key for key in missing if key not in ignore]
    if missing or unexpected:
        raise RuntimeError(
            f"{what} does not match {type(module).__name__}: "
            f"missing keys {missing}, unexpected keys {list(unexpected)}"
        )
    return module


def find_checkpoint_file(
    filename: str, *search_dirs: PathLike, required: bool = True
) -> Optional[Path]:
    """Return ``filename`` from the first of ``search_dirs`` that has it."""
    for directory in search_dirs:
        candidate = Path(directory) / filename
        if candidate.is_file():
            return candidate
    if required:
        searched = ", ".join(str(Path(d)) for d in search_dirs)
        raise FileNotFoundError(f"{filename} not found in: {searched}")
    return None


def pick_t3_checkpoint(filenames: Sequence[str], repo_id: str) -> str:
    """Pick the single ``t3_*.safetensors`` file a language-pack repo ships."""
    candidates = sorted(
        f for f in filenames if f.startswith("t3_") and f.endswith(".safetensors")
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Cannot pick a T3 checkpoint in '{repo_id}' automatically "
            f"(found {candidates}); pass t3_model=<filename> explicitly."
        )
    return candidates[0]
