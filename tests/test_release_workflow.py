"""Exercise the actual publishing step with a local GitHub CLI stand-in."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


@pytest.fixture
def publish_step(tmp_path):
    workflow = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / ".github/workflows/release.yml").read_text()
    )
    script = next(step["run"] for step in workflow["jobs"]["release"]["steps"]
                  if step.get("name") == "Create or update the draft, then publish when requested")
    binary = tmp_path / "bin"
    binary.mkdir()
    gh = binary / "gh"
    gh.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
args = sys.argv[1:]
with open(os.environ['GH_CALLS'], 'a') as f:
    f.write(json.dumps(args) + '\\n')
if args[0] == 'api':
    print('b' * 40 if os.environ.get('MOVED_TAG') == 'true' else os.environ['SOURCE_SHA'])
elif args[:2] == ['release', 'view']:
    if 'isDraft' in args:
        state = os.environ['EXISTING_RELEASE']
        if state == 'missing':
            sys.exit(1)
        print('true' if state == 'draft' else 'false')
    else:
        print('https://github.com/owner/fork/releases/tag/v0.1.7.post1')
elif args[:2] not in [['release', 'create'], ['release', 'upload'], ['release', 'edit']]:
    sys.exit('Unexpected gh command')
''')
    gh.chmod(0o755)
    (tmp_path / "asset.whl").write_bytes(b"verified asset")

    def run(existing="missing", publish=False, prerelease=False, moved=False):
        environment = {
            **os.environ,
            "PATH": str(binary) + os.pathsep + os.environ["PATH"],
            "GH_CALLS": str(tmp_path / "calls.jsonl"),
            "GITHUB_REPOSITORY": "owner/fork",
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
            "SOURCE_SHA": "a" * 40,
            "RELEASE_TAG": "v0.1.7.post1",
            "IS_PRERELEASE": str(prerelease).lower(),
            "PUBLISH": str(publish).lower(),
            "EXISTING_RELEASE": existing,
            "MOVED_TAG": str(moved).lower(),
        }
        result = subprocess.run(["bash", "-e", "-o", "pipefail", "-c", script],
                                cwd=tmp_path, env=environment, capture_output=True, text=True)
        calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
        return result, calls
    return run


@pytest.mark.parametrize("existing", ["missing", "draft", "published"])
@pytest.mark.parametrize("publish", [False, True])
def test_release_creation_retries_and_publishing(publish_step, existing, publish):
    result, calls = publish_step(existing=existing, publish=publish)
    assert result.returncode == 0, result.stderr
    writes = [call for call in calls if call[:2] in
              [["release", "create"], ["release", "upload"], ["release", "edit"]]]
    if existing == "published":
        assert writes == []
        assert "already published" in result.stdout
        return
    if existing == "missing":
        assert writes[0][:2] == ["release", "create"]
        assert "--draft" in writes[0]
        assert "--verify-tag" in writes[0]
        assert writes[0][writes[0].index("--target") + 1] == "a" * 40
    else:
        assert writes[0][:2] == ["release", "upload"]
        assert "--clobber" in writes[0]
    if publish:
        assert writes[1][:2] == ["release", "edit"]
        assert "--draft=false" in writes[1]
    else:
        assert len(writes) == 1


@pytest.mark.parametrize("prerelease", [False, True])
def test_prerelease_flag_is_passed_to_github(publish_step, prerelease):
    result, calls = publish_step(publish=True, prerelease=prerelease)
    assert result.returncode == 0, result.stderr
    create = next(call for call in calls if call[:2] == ["release", "create"])
    edit = next(call for call in calls if call[:2] == ["release", "edit"])
    assert ("--prerelease" in create) is prerelease
    assert f"--prerelease={str(prerelease).lower()}" in edit


def test_moved_tag_prevents_all_release_writes(publish_step):
    result, calls = publish_step(publish=True, moved=True)
    assert result.returncode != 0
    assert "tag moved" in result.stderr
    assert len(calls) == 1
    assert calls[0][0] == "api"
