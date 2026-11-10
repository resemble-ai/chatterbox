"""Tests for chatterbox.models.checkpoints (no model weights needed).

The module is loaded from its file path on purpose: importing the
``chatterbox`` package pulls in the full model stack (librosa, transformers,
perth, ...), which these pure loading rules do not need.
"""
import importlib.util
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

_MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "chatterbox" / "models" / "checkpoints.py"
)
_spec = importlib.util.spec_from_file_location("chatterbox_checkpoints", _MODULE_PATH)
ckpt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ckpt)

ALIASES = {"v2": "s3gen.pt", "v3": "s3gen_v3.pt"}


class Tiny(torch.nn.Module):
    ignore_state_dict_missing = ("derived",)

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.register_buffer("derived", torch.zeros(2))


# --- resolve_checkpoint_name -------------------------------------------------

def test_resolve_none_uses_default():
    assert ckpt.resolve_checkpoint_name(None, ALIASES, "s3gen.pt") == "s3gen.pt"


def test_resolve_alias():
    assert ckpt.resolve_checkpoint_name("v3", ALIASES, "s3gen.pt") == "s3gen_v3.pt"


@pytest.mark.parametrize("name", ["custom.pt", "custom.safetensors"])
def test_resolve_explicit_filename_passes_through(name):
    assert ckpt.resolve_checkpoint_name(name, ALIASES, "s3gen.pt") == name


def test_resolve_rejects_unknown_selector():
    with pytest.raises(ValueError, match="Unknown S3Gen model 'v9'"):
        ckpt.resolve_checkpoint_name("v9", ALIASES, "s3gen.pt", what="S3Gen model")


# --- load_state_dict_file ----------------------------------------------------

def test_load_state_dict_file_reads_both_formats(tmp_path):
    state = {"linear.weight": torch.ones(2, 2), "linear.bias": torch.zeros(2)}
    save_file(state, str(tmp_path / "m.safetensors"))
    torch.save(state, tmp_path / "m.pt")

    for name in ("m.safetensors", "m.pt"):
        loaded = ckpt.load_state_dict_file(tmp_path / name, map_location="cpu")
        assert set(loaded) == set(state)
        assert torch.equal(loaded["linear.weight"], state["linear.weight"])


# --- load_state_dict_checked -------------------------------------------------

def _weights_only():
    return {"linear.weight": torch.ones(2, 2), "linear.bias": torch.zeros(2)}


def test_checked_load_tolerates_declared_missing_keys():
    module = Tiny()
    # A strict load of a checkpoint without the derived buffer fails...
    with pytest.raises(RuntimeError):
        module.load_state_dict(_weights_only(), strict=True)
    # ...but the checked load accepts it when the key is declared ignorable.
    ckpt.load_state_dict_checked(
        module, _weights_only(), ignore_missing=module.ignore_state_dict_missing
    )
    assert torch.equal(module.linear.weight, torch.ones(2, 2))


def test_checked_load_rejects_undeclared_missing_keys():
    module = Tiny()
    state = _weights_only()
    del state["linear.bias"]
    with pytest.raises(RuntimeError, match=r"missing keys \['linear.bias'\]"):
        ckpt.load_state_dict_checked(
            module, state, ignore_missing=module.ignore_state_dict_missing, what="S3Gen"
        )


def test_checked_load_rejects_unexpected_keys():
    module = Tiny()
    state = _weights_only()
    state["stray"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match=r"unexpected keys \['stray'\]"):
        ckpt.load_state_dict_checked(
            module, state, ignore_missing=module.ignore_state_dict_missing
        )


# --- find_checkpoint_file ----------------------------------------------------

def test_find_prefers_first_dir_then_falls_back(tmp_path):
    pack = tmp_path / "pack"
    base = tmp_path / "base"
    pack.mkdir()
    base.mkdir()
    (pack / "t3_pt_br.safetensors").write_bytes(b"")
    (base / "ve.pt").write_bytes(b"")
    (base / "t3_pt_br.safetensors").write_bytes(b"")

    assert ckpt.find_checkpoint_file("t3_pt_br.safetensors", pack, base) == pack / "t3_pt_br.safetensors"
    assert ckpt.find_checkpoint_file("ve.pt", pack, base) == base / "ve.pt"


def test_find_missing_file(tmp_path):
    assert ckpt.find_checkpoint_file("conds.pt", tmp_path, required=False) is None
    with pytest.raises(FileNotFoundError, match="conds.pt not found in"):
        ckpt.find_checkpoint_file("conds.pt", tmp_path)


# --- pick_t3_checkpoint ------------------------------------------------------

def test_pick_t3_checkpoint_single_match():
    files = [".gitattributes", "README.md", "s3gen_v3.pt", "s3gen_v3.safetensors", "t3_pt_br.safetensors"]
    assert ckpt.pick_t3_checkpoint(files, "org/pack") == "t3_pt_br.safetensors"


@pytest.mark.parametrize(
    "files",
    [
        ["README.md", "s3gen_v3.pt"],
        ["t3_a.safetensors", "t3_b.safetensors"],
    ],
)
def test_pick_t3_checkpoint_needs_exactly_one(files):
    with pytest.raises(ValueError, match="pass t3_model=<filename> explicitly"):
        ckpt.pick_t3_checkpoint(files, "org/pack")
