"""Archive complete Hugging Face model snapshots and verify private mirrors."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path, PurePosixPath
import platform
import re
import shutil

from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError

BACKUP_DIR = "_chatterbox_backup"
MANIFEST = f"{BACKUP_DIR}/manifest.json"
CHECKSUMS = f"{BACKUP_DIR}/SHA256SUMS"


def _path(root, name):
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or ".." in relative.parts
            or "\\" in name or ":" in name or relative.as_posix() != name):
        raise ValueError(f"Unsafe archive path: {name!r}")
    path = Path(root).joinpath(*relative.parts)
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f"Archive path escapes its directory: {name!r}")
    if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root):
        raise ValueError(f"Archive paths must not be symlinks: {name!r}")
    return path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(root):
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(root).parts
    }


def archive(repo_id, output, *, revision="main", source_revision,
            source_repo="https://github.com/pbsladek/chatterbox", license_file="LICENSE"):
    """Download every upstream file at one resolved commit, then record hashes."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("source_revision must be a full 40-character Git commit SHA")
    license_file = Path(license_file)
    if not license_file.is_file():
        raise ValueError("Pass --license-file pointing to the library's LICENSE")
    output = Path(output).resolve()
    api = HfApi()
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    if not info.sha or not re.fullmatch(r"[0-9a-f]{40}", info.sha):
        raise ValueError("The Hub did not return an immutable model commit SHA")
    names = {entry.rfilename for entry in info.siblings}
    if not names:
        raise ValueError("The source repository is empty")
    for name in names:
        _path(output, name)
        if PurePosixPath(name).parts[0] in {BACKUP_DIR, ".cache", ".git"}:
            raise ValueError(f"Upstream uses a reserved archive path: {name}")
    if (output / MANIFEST).exists():
        previous = verify(output)
        if (previous["upstream"] != {"repo_id": repo_id, "revision": info.sha}
                or previous["source"]["revision"] != source_revision):
            raise ValueError("Use a new output directory for a different snapshot or source revision")
        return previous
    if output.exists() and _files(output) - names:
        raise ValueError("Output contains files outside this snapshot; use a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    # local_dir creates materialized files, suitable for an independent backup.
    snapshot_download(repo_id=repo_id, revision=info.sha, local_dir=output, repo_type="model")
    if _files(output) != names:
        raise ValueError("Downloaded inventory differs from the upstream snapshot")
    records = []
    for entry in sorted(info.siblings, key=lambda item: item.rfilename):
        path = _path(output, entry.rfilename)
        digest = sha256(path)
        if entry.size is not None and path.stat().st_size != entry.size:
            raise ValueError(f"Upstream size mismatch: {entry.rfilename}")
        if entry.lfs is not None and digest != entry.lfs.sha256:
            raise ValueError(f"Upstream SHA-256 mismatch: {entry.rfilename}")
        records.append({"path": entry.rfilename, "size": path.stat().st_size, "sha256": digest})
    backup = output / BACKUP_DIR
    backup.mkdir()
    shutil.copyfile(license_file, backup / "LIBRARY_LICENSE.txt")
    for optional in ("uv.lock", "pyproject.toml"):
        if Path(optional).is_file():
            shutil.copyfile(optional, backup / optional)
    for path in sorted(backup.iterdir()):
        records.append({"path": path.relative_to(output).as_posix(),
                        "size": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "upstream": {"repo_id": repo_id, "revision": info.sha},
        "source": {"repository": source_repo, "revision": source_revision},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()}),
        },
        "files": records,
    }
    (output / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    sums = {record["path"]: record["sha256"] for record in records}
    sums[MANIFEST] = sha256(output / MANIFEST)
    (output / CHECKSUMS).write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items())), encoding="utf-8"
    )
    verify(output)
    return manifest


def verify(directory):
    """Verify the complete inventory without contacting the network."""
    root = Path(directory).resolve()
    manifest = json.loads(_path(root, MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported archive manifest version")
    records = manifest["files"]
    names = [record["path"] for record in records]
    if (len(names) != len(set(names)) or not names
            or {MANIFEST, CHECKSUMS}.intersection(names)):
        raise ValueError("Invalid archive inventory")
    for record in records:
        path = _path(root, record["path"])
        if not path.is_file() or path.stat().st_size != record["size"]:
            raise ValueError(f"Missing file or size mismatch: {record['path']}")
        if sha256(path) != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {record['path']}")
    expected = set(names) | {MANIFEST, CHECKSUMS}
    if _files(root) != expected:
        raise ValueError("Archive contains missing or unexpected files")
    sums = {record["path"]: record["sha256"] for record in records}
    sums[MANIFEST] = sha256(root / MANIFEST)
    expected_checksums = "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items()))
    if _path(root, CHECKSUMS).read_text(encoding="utf-8") != expected_checksums:
        raise ValueError("Checksum inventory does not match the manifest")
    return manifest


def upload(directory, repo_id):
    """Upload only a verified archive, refusing existing public repositories."""
    root = Path(directory).resolve()
    manifest = verify(root)
    if repo_id.casefold() == manifest["upstream"]["repo_id"].casefold():
        raise ValueError("The mirror must be different from the upstream repository")
    api = HfApi()
    try:
        info = api.model_info(repo_id)
    except RepositoryNotFoundError:
        api.create_repo(repo_id=repo_id, repo_type="model", private=True)
        info = api.model_info(repo_id)
    if not info.private:
        raise ValueError("Refusing to upload to a public repository; choose a private mirror")
    names = {record["path"] for record in manifest["files"]} | {MANIFEST, CHECKSUMS}
    existing = set(api.list_repo_files(repo_id, repo_type="model"))
    if existing - names - {".gitattributes"}:
        raise ValueError("Mirror contains unrelated files; choose a fresh private repository")
    result = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        parent_commit=info.sha,
        commit_message=f"Archive {manifest['upstream']['repo_id']} at {manifest['upstream']['revision']}",
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=_path(root, name))
                    for name in sorted(names)],
    )
    return result.oid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("archive", help="Download a complete, pinned model snapshot")
    download.add_argument("repo_id")
    download.add_argument("output", type=Path)
    download.add_argument("--revision", default="main")
    download.add_argument("--source-revision", required=True)
    download.add_argument("--source-repo", default="https://github.com/pbsladek/chatterbox")
    download.add_argument("--license-file", default="LICENSE", type=Path)
    check = commands.add_parser("verify", help="Verify a local archive without network access")
    check.add_argument("directory", type=Path)
    push = commands.add_parser("upload", help="Upload a verified archive to a private HF repository")
    push.add_argument("directory", type=Path)
    push.add_argument("repo_id")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "archive":
            manifest = archive(**args)
            print(f"Archived {manifest['upstream']['repo_id']} at {manifest['upstream']['revision']}")
        elif command == "verify":
            manifest = verify(**args)
            print(f"Verified {len(manifest['files'])} files")
        else:
            revision = upload(**args)
            print(f"Uploaded private mirror {args['repo_id']} at {revision}")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
