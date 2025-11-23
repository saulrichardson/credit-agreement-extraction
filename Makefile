PYTHON ?= python3
POETRY ?= poetry

default: help

.PHONY: help
help:
	@echo "Targets:"
	@echo "  make install    Install deps via Poetry"
	@echo "  make lint       Run ruff"
	@echo "  make format     Run black"
	@echo "  make test       Run pytest (if any)"

.PHONY: install
install:
	$(POETRY) install

.PHONY: lint
lint:
	$(POETRY) run ruff check src

.PHONY: format
format:
	$(POETRY) run black src

.PHONY: test
test:
	@if [ -d tests ]; then $(POETRY) run pytest; else echo "No tests directory"; fi
