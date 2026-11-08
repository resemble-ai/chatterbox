"""Validate and assemble the artifacts published by GitHub Releases."""

import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import zipfile

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 test matrix
    import tomli as tomllib

from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename
from packaging.version import Version

LOCK_FILES = ("uv.lock", "models.lock.json", "mirrors.lock.json")


def project_metadata(root):
    with (Path(root) / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    version = Version(project["version"])
    return {
        "name": project["name"],
        "version": project["version"],
        "prerelease": version.is_prerelease or version.is_devrelease,
    }


def check_tag(tag, version):
    expected = f"v{version}"
    if tag != expected:
        raise ValueError(f"Tag {tag!r} does not match pyproject.toml; expected {expected!r}")
    # Validate before using the tag in output files or CLI arguments.
    Version(version)
    if not re.fullmatch(r"v[0-9][A-Za-z0-9.!+_-]*", tag):
        raise ValueError(f"Invalid release tag: {tag!r}")


def git_info(root):
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, text=True, capture_output=True
        ).stdout.strip()
    return {
        "commit": git("rev-parse", "HEAD"),
        "source_date_epoch": int(git("show", "-s", "--format=%ct", "HEAD")),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
    }


def check_distributions(root, dist):
    project = project_metadata(root)
    name, version = canonicalize_name(project["name"]), Version(project["version"])
    # uv creates this bookkeeping file in a fresh output directory.
    files = sorted(path for path in Path(dist).iterdir() if path.name != ".gitignore")
    wheels = [path for path in files if path.suffix == ".whl"]
    sources = [path for path in files if path.name.endswith(".tar.gz")]
    if len(files) != 2 or len(wheels) != 1 or len(sources) != 1:
        raise ValueError("dist must contain exactly one wheel and one .tar.gz source distribution")
    if any(path.is_symlink() or not path.is_file() for path in files):
        raise ValueError("Distribution artifacts must be regular files")
    wheel_name, wheel_version, _, _ = parse_wheel_filename(wheels[0].name)
    source_name, source_version = parse_sdist_filename(sources[0].name)
    if (wheel_name, wheel_version) != (name, version) or (source_name, source_version) != (name, version):
        raise ValueError("Distribution filenames do not match the project name and version")
    with zipfile.ZipFile(wheels[0]) as wheel:
        metadata_paths = [path for path in wheel.namelist() if path.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError("Wheel must contain one METADATA file")
        wheel_metadata = wheel.read(metadata_paths[0])
    with tarfile.open(sources[0]) as source:
        metadata_paths = [member for member in source.getmembers()
                          if member.name.count("/") == 1 and member.name.endswith("/PKG-INFO")]
        if len(metadata_paths) != 1 or not metadata_paths[0].isfile():
            raise ValueError("Source distribution must contain one top-level PKG-INFO file")
        with source.extractfile(metadata_paths[0]) as stream:
            source_metadata = stream.read()
    for data in (wheel_metadata, source_metadata):
        metadata = BytesParser().parsebytes(data)
        if (canonicalize_name(metadata.get("Name", "")), Version(metadata.get("Version", "0"))) != (name, version):
            raise ValueError("Distribution metadata does not match the project name and version")
    return files


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(root, dist, output, tag=None):
    root, output = Path(root), Path(output)
    project = project_metadata(root)
    if tag:
        check_tag(tag, project["version"])
    source = git_info(root)
    if tag and source["dirty"]:
        raise ValueError("Commit all source changes before preparing a tagged release")
    files = check_distributions(root, dist)
    for name in LOCK_FILES:
        if not (root / name).is_file():
            raise ValueError(f"Missing release lock file: {name}")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Release output must be empty to avoid publishing stale assets")
    output.mkdir(parents=True, exist_ok=True)
    for path in [*files, *(root / name for name in LOCK_FILES)]:
        shutil.copyfile(path, output / path.name)
    manifest = {"schema_version": 1, **project, **source, "tag": tag}
    (output / "release.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksums = "".join(f"{sha256(path)}  {path.name}\n" for path in sorted(output.iterdir()))
    (output / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    return manifest


def verify(directory):
    directory = Path(directory)
    expected = {}
    for line in (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._+-]+)", line)
        if not match or match[2] in expected or match[2] in {".", "..", "SHA256SUMS"}:
            raise ValueError("Invalid release checksum inventory")
        expected[match[2]] = match[1]
    if not set(LOCK_FILES).union({"release.json"}).issubset(expected):
        raise ValueError("Release checksum inventory is missing provenance or locks")
    if {path.name for path in directory.iterdir()} != set(expected).union({"SHA256SUMS"}):
        raise ValueError("Release assets contain missing or unexpected files")
    for name, digest in expected.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Release asset checksum mismatch: {name}")
    return json.loads((directory / "release.json").read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Validate a tag before running release jobs")
    check.add_argument("--tag", required=True)
    check.add_argument("--root", type=Path, default=Path.cwd())
    check.add_argument("--github-output", type=Path)
    bundle = commands.add_parser("prepare", help="Validate distributions and assemble release assets")
    bundle.add_argument("--root", type=Path, default=Path.cwd())
    bundle.add_argument("--dist", type=Path, default=Path("dist"))
    bundle.add_argument("--output", type=Path, default=Path("artifacts/release"))
    bundle.add_argument("--tag")
    integrity = commands.add_parser("verify", help="Verify downloaded release assets on any OS")
    integrity.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            result = project_metadata(args.root)
            check_tag(args.tag, result["version"])
            result.update(git_info(args.root))
            if args.github_output:
                with args.github_output.open("a", encoding="utf-8") as stream:
                    stream.write(f"version={result['version']}\n")
                    stream.write(f"prerelease={str(result['prerelease']).lower()}\n")
                    stream.write(f"sha={result['commit']}\n")
        elif args.command == "prepare":
            result = prepare(args.root, args.dist, args.output, args.tag)
        else:
            result = verify(args.directory)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError,
            tarfile.TarError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
