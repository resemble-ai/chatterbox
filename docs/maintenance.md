# Maintaining this fork

This repository builds the inference library. Existing checkpoints remain
usable; building the package does not train or download models. Model files
live separately in Hugging Face repositories or local backup directories.

## Install and validate

Use Python 3.11 as the default maintenance environment. The package declares
Python 3.10 through 3.14, and CI targets those versions on Linux plus 3.11
and 3.14 on Apple Silicon macOS. Python 3.15 is outside this baseline.

```sh
uv sync --locked --python 3.11
make check
```

`make check` runs lint, network-free tests, a wheel and source distribution
build, and metadata validation. Tests cover snapshot integrity, mirror
selection, offline tokenization, cached transformer sampling, and the bundled
Perth watermarker. CI installs the built wheel before testing and imports it
from outside the checkout. It does not download TTS checkpoints.

The core install excludes Gradio. For the demo applications:

```sh
uv sync --locked --extra demo
uv run --no-sync python gradio_tts_app.py
```

Torch and torchaudio are pinned as matching pairs: 2.6.0 for Python 3.10–3.13,
and 2.9.1 for 3.14. Perth uses the published 1.0.1 wheel, which contains its
watermark checkpoints, instead of a moving Git branch. That release requires
`pkg_resources`, so setuptools is constrained below 81. The lock includes
transitive dependencies and package hashes. These are a compatibility
baseline; dependency updates still need review and synthesis checks.

Examples save audio with SoundFile, which avoids torchaudio's newer
TorchCodec/FFmpeg requirement for file export.

Linux installations may include CUDA runtime wheels from the default PyTorch
packages. CUDA and MPS synthesis need validation on the target hardware;
passing CPU checks alone does not establish accelerator compatibility.

## Build and install your own copies

```sh
make build
uv run --no-sync twine check --strict dist/*
```

The results are `dist/chatterbox_tts-0.1.7.post1-py3-none-any.whl` and
`dist/chatterbox_tts-0.1.7.post1.tar.gz`. The wheel includes the library and
license notices. The source distribution also includes tests, scripts,
documentation, and both locks. Install a wheel in a separate environment:

```sh
uv venv --python 3.11 /tmp/chatterbox-wheel-test
uv pip install --python /tmp/chatterbox-wheel-test/bin/python dist/*.whl
```

A wheel's metadata specifies dependency requirements; installing it with pip
does not recreate `uv.lock`. Keep the source checkout and lock with every
release. For long-term preservation, also keep the Python installer and a
wheelhouse for each OS, architecture, Python version, and CUDA configuration
you use. Include the Perth wheel so its embedded models survive upstream
changes. Do not assume one machine's wheelhouse works on another platform.

The package keeps the upstream distribution name for compatibility. Install
only one implementation of `chatterbox-tts` in an environment. In a consumer,
pin the fork to a full commit SHA or to a wheel with a recorded SHA-256:

```text
chatterbox-tts @ git+https://github.com/pbsladek/chatterbox.git@FULL_FORK_COMMIT_SHA
```

Yakbox currently has its own dependency overrides. Pointing it at this fork
does not apply this repository's lock. Align those overrides deliberately or
run inference in a separate environment, then repeat the audio checks.

## Preserve checkpoints

`models.lock.json` records immutable upstream commits, file sizes, available
LFS SHA-256 values, and model card license declarations. The Original and
Multilingual variants share a repository. Turbo and Nano each have their own.
The original repository also contains older checkpoints, so a complete
snapshot uses more space than a single inference variant.

| Source repository | Recorded revision | Complete snapshot |
| --- | --- | --- |
| `ResembleAI/chatterbox` | `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` | about 12.91 GiB |
| `ResembleAI/chatterbox-turbo` | `749d1c1a46eb10492095d68fbcf55691ccf137cd` | about 3.77 GiB |
| `ResembleAI/chatterbox-nano` | `71ccd1d0081b430592cea481f4307e764e07bc64` | about 2.79 GiB |

Commit the source changes you want to preserve, then record that full source
SHA when archiving. Run from the repository root so the tool can copy the
library license, `pyproject.toml`, and `uv.lock` alongside the model snapshot:

```sh
uv run --no-sync chatterbox-models archive ResembleAI/chatterbox \
  model-archives/chatterbox \
  --revision 5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18 \
  --source-revision "$(git rev-parse HEAD)"

uv run --no-sync chatterbox-models archive ResembleAI/chatterbox-turbo \
  model-archives/chatterbox-turbo \
  --revision 749d1c1a46eb10492095d68fbcf55691ccf137cd \
  --source-revision "$(git rev-parse HEAD)"

uv run --no-sync chatterbox-models archive ResembleAI/chatterbox-nano \
  model-archives/chatterbox-nano \
  --revision 71ccd1d0081b430592cea481f4307e764e07bc64 \
  --source-revision "$(git rev-parse HEAD)"
```

The tool downloads every file from one resolved model commit, including
tokenizers, conditioning files, configuration, model cards, and any upstream
license files. It checks upstream sizes and LFS hashes, then records local
SHA-256 hashes, the source reference, and installed package versions under
`_chatterbox_backup/`. A source SHA is a reference to the commit you supply;
the tool does not commit your changes or archive the Git repository. Keep
your source release or a Git bundle independently.

Verify a backup without network access:

```sh
uv run --no-sync chatterbox-models verify model-archives/chatterbox
```

Copy each complete directory to an independent disk or backup service and
verify that copy too. Hugging Face's download cache is not an independent
backup. Download bookkeeping under `.cache/` is excluded from the backup
inventory and uploads.

## Upload private mirrors under pbsladek

The initial private mirrors have been uploaded and their remote inventories,
sizes, Git blob IDs, and LFS SHA-256 values verified against the local backups.
Their immutable commits are recorded in `mirrors.lock.json`:

| Mirror | Commit |
| --- | --- |
| `pbsladek/chatterbox` | `94dd991b62f6a6cb513ee082db46714cf923d9bb` |
| `pbsladek/chatterbox-turbo` | `40b0bb1b9d7f428d58d65849e85cb253fb26c5bf` |
| `pbsladek/chatterbox-nano` | `7408d84af552de9456ec80bf94482c293b1cf181` |

These initial backups reference upstream source commit
`ab55923c27757c90cfcc407baa57dc25a7d25217`. The maintenance changes in this
checkout still need their own commit and release; the built source
distribution preserves the current fork files separately.

Authenticate locally with `hf auth login` or a configured `HF_TOKEN`. Do not
put a token in source files or shell command arguments. Then:

```sh
uv run --no-sync chatterbox-models upload model-archives/chatterbox pbsladek/chatterbox
uv run --no-sync chatterbox-models upload model-archives/chatterbox-turbo pbsladek/chatterbox-turbo
uv run --no-sync chatterbox-models upload model-archives/chatterbox-nano pbsladek/chatterbox-nano
```

The upload command verifies the archive first, creates a private model
repository if needed, and refuses existing public repositories, upstream
destinations, or repositories containing unrelated files. It uploads only
the verified inventory and prints the resulting mirror commit SHA. It does
not delete destination files or change repository visibility. Confirm storage
availability in your Hugging Face account before uploading the full set.

Pin inference to the **mirror's** resulting commit SHA. It differs from the
upstream commit because the mirror includes backup metadata:

```python
from chatterbox import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(
    device="cpu",
    repo_id="pbsladek/chatterbox",
    revision="94dd991b62f6a6cb513ee082db46714cf923d9bb",
)
```

All four loaders accept `repo_id`, `revision`, `cache_dir`,
`local_files_only`, and `token` as keyword arguments. Existing calls still
use upstream defaults. Turbo/Nano also retain `nano=True`; Multilingual
retains `t3_model="v2"` or `"v3"`. Omit `token` to use Hugging Face's normal
local authentication.

Use `local_files_only=True` for an already populated Hub cache. For a
standalone backup, bypass the Hub entirely:

```python
from chatterbox import ChatterboxTTS, ChatterboxMultilingualTTS, ChatterboxTurboTTS

model = ChatterboxTTS.from_local("model-archives/chatterbox", device="cpu")
multilingual = ChatterboxMultilingualTTS.from_local(
    "model-archives/chatterbox", device="cpu", t3_model="v3",
)
turbo = ChatterboxTurboTTS.from_local("model-archives/chatterbox-turbo", device="cpu")
nano = ChatterboxTurboTTS.from_local("model-archives/chatterbox-nano", device="cpu", nano=True)
```

The Chinese Cangjie mapping now loads from the tokenizer's own checkpoint
directory. Local multilingual inference no longer needs a separate upstream
lookup for that file.

Chinese word segmentation also uses local weights: put a previously downloaded
`spacy_ontonotes` pkuseg model under `<checkpoint-directory>/pkuseg/`, or retain
it under pkuseg's existing local cache. Without those separate weights,
tokenization keeps Cangjie conversion and skips word segmentation with a
warning. It does not download another model during construction. Preserve
those weights and their own notices in your independent backup if you use
Chinese synthesis; they are not part of the upstream Chatterbox snapshots.

## Check synthesis after changes

```sh
HF_HUB_OFFLINE=1 uv run --no-sync python scripts/smoke_inference.py \
  --variant english --model-dir model-archives/chatterbox \
  --output artifacts/english.wav

HF_HUB_OFFLINE=1 uv run --no-sync python scripts/smoke_inference.py \
  --variant turbo --model-dir model-archives/chatterbox-turbo \
  --reference /path/to/your/reference-over-5-seconds.wav \
  --output artifacts/turbo.wav
```

Use `--variant multilingual --t3-model v3 --language fr` or `--variant nano`
for the other models. For a Hub download, replace `--model-dir` with
`--repo-id` and `--revision`. The script writes audio plus a JSON report with
signal health, timing, seed, package versions, and audio checksums. Tests
reject empty, nonfinite, or silent output. A fixed seed helps comparison but
does not promise identical samples across devices or dependency versions.

Keep a reference clip you have permission to use, fixed texts, and the
generated WAV/JSON pairs in your independent backup. Listen for missing or
repeated words, voice similarity, pacing, clipping, and pronunciation. Signal
checks alone do not establish speech quality. The scripts retain the Perth
watermarking path.

## Release a version

Publishing uses GitHub Releases under `pbsladek/chatterbox`. The workflows
need no PyPI account or personal access token. The release job receives
`contents: write` through GitHub's repository token; build and test jobs have
read access.

Merge the workflows into the default branch (`master`) first. GitHub only
shows the manual **Run workflow** option for workflows on the default branch.

1. Update `project.version` in `pyproject.toml`, then run `uv lock` and
   `uv sync --locked`.
2. Run `make check` and the synthesis fixtures for the devices you support.
   Run `make release-check TAG=v0.1.7.post1`, substituting your actual version.
   The tag must exactly match `v` followed by the package version.
3. Commit and push the source, workflow files, and locks. Then create and push
   an annotated tag:

   ```sh
   git tag -a v0.1.7.post1 -m "Chatterbox fork 0.1.7.post1"
   git push origin v0.1.7.post1
   ```

The **Release** workflow resolves that tag to a commit and checks its version
before installing inference dependencies. It calls the same CI workflow used
for branches and pull requests. CI builds one wheel and source distribution,
then tests that exact wheel on Python 3.10–3.14 on Linux and Python 3.11/3.14
on Apple Silicon macOS. A failed build or test prevents release creation.

After every check passes, the workflow creates a draft containing:

- The Python wheel and source distribution.
- `uv.lock`, `models.lock.json`, and `mirrors.lock.json`.
- `release.json`, recording the package version and source commit.
- `SHA256SUMS`, covering all the other assets.

Review the draft under **Releases**, then choose **Publish release**. To
publish through Actions instead, open **Actions → Release → Run workflow**,
enter the existing tag, and check **Publish the release after every check
passes**. Leaving that option unchecked creates or refreshes a draft.

The equivalent command for a checked tag is:

```sh
gh workflow run release.yml --repo pbsladek/chatterbox --ref master \
  -f tag=v0.1.7.post1 -f publish=true
```

Manual publishing rebuilds and tests the tagged source before publishing.
Prerelease and development versions, such as `0.2.0rc1` and `0.2.0.dev1`,
become GitHub prereleases. Post releases such as `0.1.7.post1` are stable
releases.

A retry updates an existing draft's assets. An already published release is
left unchanged. The workflow also checks that the remote tag still points to
the tested commit immediately before writing release assets. Create a new
version for changes to a published release instead of moving its tag.

You can inspect a downloaded release bundle on either Linux or macOS:

```sh
uv run --no-sync python scripts/release.py verify /path/to/release-assets
```

The workflows do not publish to PyPI, download TTS checkpoints, or upload model
weights. Keep the source release, regression audio, and independent model
backups together for long-term maintenance.

## References and notices

The backup and mirror commands use the official Hugging Face
[snapshot download](https://huggingface.co/docs/huggingface_hub/guides/download)
and [upload APIs](https://huggingface.co/docs/huggingface_hub/guides/upload).
The [Original](https://huggingface.co/ResembleAI/chatterbox),
[Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo), and
[Nano](https://huggingface.co/ResembleAI/chatterbox-nano) model cards declare
MIT licensing at the recorded revisions. Preserve those model cards and all
notices. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for the
library's bundled source attributions. Review independently published
conversions and fine-tunes under their own notices.
