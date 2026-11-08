import importlib.util
import io
import json
from pathlib import Path
import tarfile
import zipfile

import pytest

spec = importlib.util.spec_from_file_location(
    "release_tools", Path(__file__).resolve().parents[1] / "scripts/release.py"
)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "chatterbox-tts"\nversion = "0.1.7.post1"\n'
    )
    for name in release.LOCK_FILES:
        (tmp_path / name).write_text(f"test lock: {name}\n")
    metadata = b"Metadata-Version: 2.4\nName: chatterbox-tts\nVersion: 0.1.7.post1\n"
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / ".gitignore").write_text("*")
    with zipfile.ZipFile(dist / "chatterbox_tts-0.1.7.post1-py3-none-any.whl", "w") as wheel:
        wheel.writestr("chatterbox_tts-0.1.7.post1.dist-info/METADATA", metadata)
    with tarfile.open(dist / "chatterbox_tts-0.1.7.post1.tar.gz", "w:gz") as source:
        member = tarfile.TarInfo("chatterbox_tts-0.1.7.post1/PKG-INFO")
        member.size = len(metadata)
        source.addfile(member, io.BytesIO(metadata))
    monkeypatch.setattr(release, "git_info", lambda root: {
        "commit": "a" * 40, "source_date_epoch": 1700000000, "dirty": False,
    })
    return tmp_path


def test_tag_must_exactly_match_package_version():
    release.check_tag("v0.1.7.post1", "0.1.7.post1")
    for invalid in ["0.1.7.post1", "v0.1.7", "master", "v0.1.7.post1\nsha=wrong"]:
        with pytest.raises(ValueError):
            release.check_tag(invalid, "0.1.7.post1")


@pytest.mark.parametrize("version,expected", [
    ("0.1.7.post1", False), ("0.2.0", False), ("0.2.0rc1", True), ("0.2.0.dev1", True),
])
def test_prerelease_detection_uses_python_version_rules(tmp_path, version, expected):
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname="fork"\nversion="{version}"\n')
    assert release.project_metadata(tmp_path)["prerelease"] is expected


def test_prepare_preserves_exact_distributions_locks_and_source_provenance(project):
    output = project / "release"
    result = release.prepare(project, project / "dist", output, "v0.1.7.post1")
    assert result["commit"] == "a" * 40
    assert result["tag"] == "v0.1.7.post1"
    assert result["dirty"] is False
    assert release.verify(output) == result
    assert ".gitignore" not in {path.name for path in output.iterdir()}
    assert len(list(output.iterdir())) == 7
    for name in release.LOCK_FILES:
        assert (output / name).read_bytes() == (project / name).read_bytes()


def test_tagged_release_rejects_dirty_source(project, monkeypatch):
    monkeypatch.setattr(release, "git_info", lambda root: {"dirty": True})
    with pytest.raises(ValueError, match="Commit all source changes"):
        release.prepare(project, project / "dist", project / "release", "v0.1.7.post1")


def test_wrong_tag_fails_before_build_artifacts_are_read(project):
    with pytest.raises(ValueError, match="does not match"):
        release.prepare(project, project / "missing", project / "release", "v9.0.0")


def test_stale_distribution_is_rejected(project):
    (project / "dist/stale.whl").write_bytes(b"old artifact")
    with pytest.raises(ValueError, match="exactly one wheel"):
        release.check_distributions(project, project / "dist")


def test_wheel_metadata_must_match_its_filename_and_project(project):
    wheel = next((project / "dist").glob("*.whl"))
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("fork.dist-info/METADATA", "Name: other-project\nVersion: 0.1.7.post1\n")
    with pytest.raises(ValueError, match="metadata does not match"):
        release.check_distributions(project, project / "dist")


def test_incomplete_locks_or_nonempty_output_are_rejected(project):
    output = project / "release"
    output.mkdir()
    (output / "stale").touch()
    with pytest.raises(ValueError, match="output must be empty"):
        release.prepare(project, project / "dist", output)
    (project / "mirrors.lock.json").unlink()
    with pytest.raises(ValueError, match="Missing release lock"):
        release.prepare(project, project / "dist", project / "another-release")


@pytest.mark.parametrize("change", ["corrupt", "missing", "extra", "symlink", "path_traversal"])
def test_verify_detects_invalid_release_assets(project, change):
    output = project / "release"
    release.prepare(project, project / "dist", output)
    path = output / "uv.lock"
    if change == "corrupt":
        path.write_text("corrupted")
    elif change == "missing":
        path.unlink()
    elif change == "extra":
        (output / "extra").touch()
    elif change == "symlink":
        path.unlink()
        path.symlink_to(project / "uv.lock")
    else:
        (output / "SHA256SUMS").write_text("a" * 64 + "  ../uv.lock\n")
    with pytest.raises(ValueError):
        release.verify(output)


def test_check_cli_writes_validated_github_outputs(project):
    output = project / "github-output"
    release.main(["check", "--root", str(project), "--tag", "v0.1.7.post1",
                  "--github-output", str(output)])
    assert output.read_text().splitlines() == [
        "version=0.1.7.post1", "prerelease=false", "sha=" + "a" * 40,
    ]
