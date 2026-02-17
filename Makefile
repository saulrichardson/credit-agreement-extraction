PYTHON ?= python3
POETRY ?= poetry

DEFAULT_RUN_ID ?= demo-recursive
DEFAULT_TARBALL ?=
DEFAULT_ITEM_IDS_FILE ?=
DEFAULT_GATEWAY_URL ?= http://127.0.0.1:8000

default: help

.PHONY: help
help:
	@echo "Targets:"
	@echo "  make install            Install deps via Poetry"
	@echo "  make lint               Run ruff"
	@echo "  make format             Run black"
	@echo "  make test               Run pytest"
	@echo "  make pipeline-help      Show pipeline CLI commands"
	@echo "  make run-all-v2-full    Run locked recursive full pipeline"

.PHONY: install
install:
	$(POETRY) install

.PHONY: lint
lint:
	$(POETRY) run ruff check src scripts

.PHONY: format
format:
	$(POETRY) run black src scripts

.PHONY: test
test:
	@if [ -d tests ]; then $(POETRY) run pytest; else echo "No tests directory"; fi

.PHONY: pipeline-help
pipeline-help:
	$(POETRY) run pipeline --help

.PHONY: run-all-v2-full
run-all-v2-full:
	@if [ -z "$(DEFAULT_TARBALL)" ]; then echo "Set DEFAULT_TARBALL=/path/to/filings.tar.gz"; exit 2; fi
	@if [ -z "$(DEFAULT_ITEM_IDS_FILE)" ]; then echo "Set DEFAULT_ITEM_IDS_FILE=/path/to/item_ids.json"; exit 2; fi
	$(POETRY) run pipeline all-v2-full \
	  --run-id $(DEFAULT_RUN_ID) \
	  --tarball $(DEFAULT_TARBALL) \
	  --item-ids-file $(DEFAULT_ITEM_IDS_FILE) \
	  --gateway-url $(DEFAULT_GATEWAY_URL)
