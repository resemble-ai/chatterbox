.PHONY: install test lint build check smoke release-check

install:
	uv sync --locked

test:
	uv run --no-sync pytest -q

lint:
	uv run --no-sync ruff check src tests scripts

build:
	uv run --no-sync python -m build

check: lint test build
	uv run --no-sync twine check --strict dist/*

# Example: make smoke MODEL_DIR=model-archives/chatterbox
smoke:
	uv run --no-sync python scripts/smoke_inference.py --model-dir "$(MODEL_DIR)"

release-check:
	uv run --no-sync python scripts/release.py check --tag "$(TAG)"
