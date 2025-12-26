.PHONY: build clean install install-dev test lint format publish publish-test help

# Default target
help:
	@echo "Available commands:"
	@echo "  make build        - Build distribution packages (wheel and sdist)"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make install      - Install the package locally"
	@echo "  make install-dev  - Install the package in editable mode with dev dependencies"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linter"
	@echo "  make format       - Format code with black"
	@echo "  make publish-test - Upload to TestPyPI"
	@echo "  make publish      - Upload to PyPI"
	@echo "  make bump-patch   - Bump patch version (0.1.0 -> 0.1.1)"
	@echo "  make bump-minor   - Bump minor version (0.1.0 -> 0.2.0)"
	@echo "  make bump-major   - Bump major version (0.1.0 -> 1.0.0)"

# Build distribution packages
build: clean
	python -m build

# Clean build artifacts
clean:
	rm -rf dist/ build/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Install the package locally
install:
	pip install .

# Install in editable mode for development
install-dev:
	pip install -e ".[mlx]"

# Run tests
test:
	python -m pytest tests/ -v

# Run linter
lint:
	python -m ruff check src/

# Format code
format:
	python -m black src/

# Upload to TestPyPI
publish-test: build
	python -m twine upload --repository testpypi dist/*

# Upload to PyPI
publish: build
	python -m twine upload dist/*

# Version bumping helpers
bump-patch:
	@current=$$(grep -o 'version = "[^"]*"' pyproject.toml | head -1 | grep -o '[0-9]*\.[0-9]*\.[0-9]*'); \
	major=$$(echo $$current | cut -d. -f1); \
	minor=$$(echo $$current | cut -d. -f2); \
	patch=$$(echo $$current | cut -d. -f3); \
	new_patch=$$((patch + 1)); \
	new_version="$$major.$$minor.$$new_patch"; \
	sed -i '' "s/version = \"$$current\"/version = \"$$new_version\"/" pyproject.toml; \
	echo "Bumped version: $$current -> $$new_version"

bump-minor:
	@current=$$(grep -o 'version = "[^"]*"' pyproject.toml | head -1 | grep -o '[0-9]*\.[0-9]*\.[0-9]*'); \
	major=$$(echo $$current | cut -d. -f1); \
	minor=$$(echo $$current | cut -d. -f2); \
	new_minor=$$((minor + 1)); \
	new_version="$$major.$$new_minor.0"; \
	sed -i '' "s/version = \"$$current\"/version = \"$$new_version\"/" pyproject.toml; \
	echo "Bumped version: $$current -> $$new_version"

bump-major:
	@current=$$(grep -o 'version = "[^"]*"' pyproject.toml | head -1 | grep -o '[0-9]*\.[0-9]*\.[0-9]*'); \
	major=$$(echo $$current | cut -d. -f1); \
	new_major=$$((major + 1)); \
	new_version="$$new_major.0.0"; \
	sed -i '' "s/version = \"$$current\"/version = \"$$new_version\"/" pyproject.toml; \
	echo "Bumped version: $$current -> $$new_version"
