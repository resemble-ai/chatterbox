# Chatterbox Streaming TTS Server
#
# Build:
#   docker build -t chatterbox-server .
#
# Run (GPU):
#   docker run --gpus all -p 8000:8000 \
#     -v hf-cache:/root/.cache/huggingface \
#     -v torch-cache:/cache \
#     chatterbox-server
#
# Run (CPU only):
#   docker run -p 8000:8000 \
#     -v hf-cache:/root/.cache/huggingface \
#     -v torch-cache:/cache \
#     chatterbox-server --model base --device cpu
#
# Two named volumes are used:
#   hf-cache     — HuggingFace model weights (downloaded once, reused across restarts)
#   torch-cache  — torch.compile kernel cache (compiled once per model, ~200-500 MB each;
#                  without this volume the ~60 s compilation runs on every container start)
#
# Extra CLI flags are forwarded to server.py, e.g.:
#   docker run ... chatterbox-server --model turbo --device cuda --debug

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# git is required for the resemble-perth pip dependency (installed from GitHub)
# ffmpeg is required by librosa for audio decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Layer 1: install Python dependencies (cached unless pyproject.toml changes)
# We copy pyproject.toml + a minimal dummy package so pip can resolve and
# install all deps without needing the full source tree.
COPY pyproject.toml .
RUN mkdir -p src/chatterbox && touch src/chatterbox/__init__.py
RUN pip install --no-cache-dir -e .

# ── Layer 1b: patch transformers' output_capturing.py ────────────────────────
#
# WHY THIS EXISTS:
#   torch.compile() traces through the model's forward() call, which is
#   decorated by transformers' internal output_capturing.py wrapper. That
#   wrapper uses `torch` at runtime but never imports it at the module level —
#   a bug in transformers that has not been fixed as of 5.5.2 (upstream issue).
#   Without this patch, compile raises:
#     NameError: name 'torch' is not defined
#       File "transformers/utils/output_capturing.py", line 222, in wrapper
#
# WHY THE PATCH IS SAFE:
#   - `torch` is always installed as a hard dependency of transformers, so it
#     is guaranteed to be importable.
#   - `import torch` is inserted after any `from __future__` lines (which Python
#     requires to be first). If torch is already imported, the patch is skipped.
#   - Re-importing an already-loaded module is a no-op (sys.modules cache).
#   - If a future transformers release adds the import itself, this patch will
#     print "already patched" and do nothing.
#
# WHEN TO REMOVE THIS:
#   Once the upstream bug is fixed, the `if 'import torch' not in src` guard
#   makes this RUN layer a no-op. You can then delete it for cleanliness, but
#   leaving it is harmless.
#
RUN PATCH_FILE=$(python3 -c "import transformers.utils.output_capturing as m; print(m.__file__)") \
    && if ! grep -q '^import torch' "$PATCH_FILE"; then \
        sed -i '/^from __future__ import annotations/a import torch' "$PATCH_FILE" \
        && echo "Applied patch: added import torch to output_capturing.py"; \
    else \
        echo "output_capturing.py: no patch needed, skipping"; \
    fi

# ── Layer 2: server dependencies (not listed in pyproject.toml)
RUN pip install --no-cache-dir "fastapi>=0.110" "uvicorn[standard]>=0.29" "python-multipart>=0.0.9"

# ── Layer 3: real source (invalidates only when source changes, not on dep bumps)
COPY src/ src/
COPY server/ server/

EXPOSE 8000

# HuggingFace model weights — mount as a named volume so downloads persist.
ENV HF_HOME=/root/.cache/huggingface

# torch.compile kernel cache — mount as a named volume (-v torch-cache:/cache) so
# compiled CUDA kernels survive container restarts. Without this, each startup
# recompiles from scratch (~60 s per model). Each model caches independently, so
# switching models compiles once and is then fast on all future restarts.
ENV TORCHINDUCTOR_CACHE_DIR=/cache/torch_inductor
ENV TRITON_CACHE_DIR=/cache/triton

# --host 0.0.0.0 is required so the port is reachable outside the container.
# CMD provides defaults that can be overridden at `docker run` time.
ENTRYPOINT ["python", "server/server.py", "--host", "0.0.0.0"]
CMD ["--model", "multilingual", "--device", "auto", "--port", "8000"]
