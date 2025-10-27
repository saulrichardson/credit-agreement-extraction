PYTHON ?= python3
POETRY ?= poetry
PYENV_VERSION ?= 3.11.9

default: help

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make install            Install dependencies via Poetry"
	@echo "  make shell              Enter Poetry shell"
	@echo "  make lint               Run Ruff"
	@echo "  make format             Run Black"
	@echo "  make test               Run pytest (if present)"
	@echo "  make requirements       Export requirements.txt via Poetry"
	@echo "  make manifest           Build manifest (requires TAR_ROOT, OUTPUT)"
	@echo "  make extract            Extract segments (MANIFEST, TAR_ROOT, OUTPUT_DIR)"
	@echo "  make normalize          Normalize segments (MANIFEST, TAR_ROOT, OUTPUT)"
	@echo "  make pipeline           Run manifest+extract (+normalize optional)"
	@echo "  make gateway            Launch gateway-server (reads .env)"

.PHONY: install
install:
	@echo ">> Installing dependencies with Poetry (PYENV_VERSION=$(PYENV_VERSION))"
	PYENV_VERSION=$(PYENV_VERSION) $(POETRY) install

.PHONY: shell
shell:
	$(POETRY) shell

.PHONY: lint
lint:
	$(POETRY) run ruff check src

.PHONY: format
format:
	$(POETRY) run black src

.PHONY: test
test:
	@if [ -d tests ]; then $(POETRY) run pytest; else echo "No tests/ directory found."; fi

.PHONY: requirements
requirements:
	PYENV_VERSION=$(PYENV_VERSION) $(POETRY) export --without-hashes --format requirements.txt --output requirements.txt

.PHONY: manifest
manifest:
	@if [ -z "$$TAR_ROOT" ] || [ -z "$$OUTPUT" ]; then \
		echo "Usage: make manifest TAR_ROOT=/path/to/tars OUTPUT=manifest.parquet"; exit 1; fi
	$(POETRY) run edgar-pipeline build-manifest --tar-root "$$TAR_ROOT" --output "$$OUTPUT" $(if $(PATTERN),--pattern $(PATTERN)) $(if $(LIMIT),--limit $(LIMIT))

.PHONY: extract
extract:
	@if [ -z "$$MANIFEST" ] || [ -z "$$TAR_ROOT" ] || [ -z "$$OUTPUT_DIR" ]; then \
		echo "Usage: make extract MANIFEST=manifest.parquet TAR_ROOT=/path OUTPUT_DIR=./segments"; exit 1; fi
	$(POETRY) run edgar-pipeline extract --manifest "$$MANIFEST" --tar-root "$$TAR_ROOT" --output-dir "$$OUTPUT_DIR" $(if $(METADATA_OUT),--metadata-out $(METADATA_OUT)) $(if $(CONVERT_HTML),--convert-html) $(if $(TABLE_MARKERS),--table-markers)

.PHONY: normalize
normalize:
	@if [ -z "$$MANIFEST" ] || [ -z "$$TAR_ROOT" ] || [ -z "$$OUTPUT" ]; then \
		echo "Usage: make normalize MANIFEST=manifest.parquet TAR_ROOT=/path OUTPUT=normalized.parquet"; exit 1; fi
	$(POETRY) run edgar-pipeline normalize --manifest "$$MANIFEST" --tar-root "$$TAR_ROOT" --output "$$OUTPUT" $(if $(LIMIT),--limit $(LIMIT))

.PHONY: pipeline
pipeline:
	@if [ -z "$$TAR_ROOT" ] || [ -z "$$MANIFEST_OUT" ] || [ -z "$$EXTRACT_DIR" ]; then \
		echo "Usage: make pipeline TAR_ROOT=/path MANIFEST_OUT=manifest.parquet EXTRACT_DIR=./segments [NORMALIZED_OUT=normalized.parquet]"; exit 1; fi
	$(POETRY) run edgar-pipeline run --tar-root "$$TAR_ROOT" --manifest-out "$$MANIFEST_OUT" --extract-dir "$$EXTRACT_DIR" $(if $(NORMALIZED_OUT),--normalized-out $(NORMALIZED_OUT)) $(if $(PATTERN),--pattern $(PATTERN)) $(if $(LIMIT),--limit $(LIMIT))

.PHONY: gateway
gateway:
	$(POETRY) run gateway-server --tar-root $${TAR_ROOT:-$$TAR_ROOT} --manifest $${MANIFEST_PATH:-$$MANIFEST_PATH} --host $${GATEWAY_HOST:-0.0.0.0} --port $${GATEWAY_PORT:-8080} $(if $(RELOAD),--reload)
