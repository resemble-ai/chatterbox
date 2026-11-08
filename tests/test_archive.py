import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import httpx

from chatterbox import archive as module

MODEL_SHA = "a" * 40
SOURCE_SHA = "b" * 40


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    files = {"README.md": b"---\nlicense: mit\n---\nModel card\n",
             "weights.safetensors": b"test weights", "nested/tokenizer.json": b"{}"}
    info = SimpleNamespace(sha=MODEL_SHA, private=True, siblings=[
        SimpleNamespace(rfilename=name, size=len(data),
                        lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()))
        for name, data in files.items()
    ])
    api = Mock()
    api.model_info.return_value = info
    api.list_repo_files.return_value = [".gitattributes"]
    api.create_commit.return_value = SimpleNamespace(oid="c" * 40)
    monkeypatch.setattr(module, "HfApi", lambda: api)
    monkeypatch.setattr(module, "distributions", lambda: [])

    def download(**kwargs):
        assert kwargs["revision"] == MODEL_SHA
        root = Path(kwargs["local_dir"])
        for name, data in files.items():
            path = root / name
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_bytes(data)
        cache = root / ".cache/huggingface"
        cache.mkdir(parents=True)
        (cache / "local-cache-metadata").write_text("must not be uploaded")

    monkeypatch.setattr(module, "snapshot_download", download)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "LICENSE").write_text("Library MIT license")
    (tmp_path / "uv.lock").write_text("test lockfile")
    root = tmp_path / "backup"
    return root, api, files


def make_archive(snapshot):
    root, api, files = snapshot
    manifest = module.archive("upstream/model", root, source_revision=SOURCE_SHA)
    return root, api, files, manifest


def test_archive_pins_one_snapshot_preserves_all_files_and_verifies_offline(snapshot, monkeypatch):
    root, api, files, manifest = make_archive(snapshot)
    assert manifest["upstream"]["revision"] == MODEL_SHA
    assert manifest["source"]["revision"] == SOURCE_SHA
    assert set(files) <= {record["path"] for record in manifest["files"]}
    assert (root / module.BACKUP_DIR / "LIBRARY_LICENSE.txt").read_text() == "Library MIT license"
    assert (root / module.BACKUP_DIR / "uv.lock").read_text() == "test lockfile"
    monkeypatch.setattr(module, "HfApi", Mock(side_effect=AssertionError("Network forbidden")))
    assert module.verify(root) == manifest


def test_archive_rejects_nonimmutable_source_revision(snapshot):
    root, api, _ = snapshot
    with pytest.raises(ValueError, match="full 40-character"):
        module.archive("upstream/model", root, source_revision="master")
    api.model_info.assert_not_called()


def test_archive_checks_upstream_weight_hash(snapshot):
    root, api, _ = snapshot
    api.model_info.return_value.siblings[0].lfs.sha256 = "0" * 64
    with pytest.raises(ValueError, match="Upstream SHA-256 mismatch"):
        module.archive("upstream/model", root, source_revision=SOURCE_SHA)


def test_existing_archive_is_verified_and_never_reused_for_another_revision(snapshot):
    root, api, _, manifest = make_archive(snapshot)
    assert module.archive("upstream/model", root, source_revision=SOURCE_SHA) == manifest
    api.model_info.return_value.sha = "d" * 40
    with pytest.raises(ValueError, match="new output directory"):
        module.archive("upstream/model", root, source_revision=SOURCE_SHA)


@pytest.mark.parametrize("change", ["corrupt", "delete", "extra", "manifest", "checksums"])
def test_verify_detects_damage_or_unexpected_files(snapshot, change):
    root, _, _, _ = make_archive(snapshot)
    if change == "corrupt":
        (root / "weights.safetensors").write_bytes(b"TEST weights")  # same size
    elif change == "delete":
        (root / "weights.safetensors").unlink()
    elif change == "extra":
        (root / "unexpected-secret").write_text("do not upload")
    elif change == "manifest":
        path = root / module.MANIFEST
        path.write_text(path.read_text() + " ")
    else:
        (root / module.CHECKSUMS).write_text("wrong")
    with pytest.raises(ValueError):
        module.verify(root)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "nested/../../escape", "C:\\escape"])
def test_verify_rejects_path_traversal(snapshot, name):
    root, _, _, _ = make_archive(snapshot)
    path = root / module.MANIFEST
    manifest = json.loads(path.read_text())
    manifest["files"][0]["path"] = name
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsafe archive path"):
        module.verify(root)


def test_verify_rejects_symlink(snapshot):
    root, _, _, _ = make_archive(snapshot)
    weights = root / "weights.safetensors"
    outside = root.parent / "external"
    weights.rename(outside)
    weights.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes|symlinks"):
        module.verify(root)


def test_upload_refuses_public_repository(snapshot):
    root, api, _, _ = make_archive(snapshot)
    api.model_info.return_value.private = False
    with pytest.raises(ValueError, match="public repository"):
        module.upload(root, "owner/mirror")
    api.create_commit.assert_not_called()


def test_upload_creates_new_repository_as_private(snapshot):
    root, api, _, _ = make_archive(snapshot)
    info = api.model_info.return_value
    response = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co/api/models/owner/new-mirror"))
    api.model_info.side_effect = [module.RepositoryNotFoundError("Not found", response=response), info]
    module.upload(root, "owner/new-mirror")
    api.create_repo.assert_called_once_with(repo_id="owner/new-mirror", repo_type="model", private=True)


def test_upload_refuses_upstream_or_unrelated_destination_files(snapshot):
    root, api, _, _ = make_archive(snapshot)
    with pytest.raises(ValueError, match="different from the upstream"):
        module.upload(root, "upstream/model")
    api.list_repo_files.return_value = ["unrelated.txt"]
    with pytest.raises(ValueError, match="unrelated files"):
        module.upload(root, "owner/mirror")
    api.create_commit.assert_not_called()


def test_upload_only_sends_verified_inventory_and_returns_mirror_commit(snapshot):
    root, api, files, _ = make_archive(snapshot)
    assert module.upload(root, "owner/mirror") == "c" * 40
    call = api.create_commit.call_args.kwargs
    names = {operation.path_in_repo for operation in call["operations"]}
    assert set(files) <= names
    assert {module.MANIFEST, module.CHECKSUMS} <= names
    assert not any(".cache" in name for name in names)
    assert call["parent_commit"] == MODEL_SHA


def test_upload_corrupt_archive_never_contacts_hub(snapshot, monkeypatch):
    root, _, _, _ = make_archive(snapshot)
    (root / "weights.safetensors").write_bytes(b"broken")
    hub = Mock()
    monkeypatch.setattr(module, "HfApi", hub)
    with pytest.raises(ValueError):
        module.upload(root, "owner/mirror")
    hub.assert_not_called()
