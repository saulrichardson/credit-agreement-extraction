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
	@echo "  make contractir-v0_2-strategy   Run ContractIR v0.2 strategy harness"
	@echo "  make contractir-v0_2-validate   Validate a ContractIR v0.2 JSON artifact"
	@echo "  make contractir-v0_2-flow      Run ContractIR v0.2 end-to-end flow"

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

# ----------------------- ContractIR v0.2 (pricing kernel) -----------------------

SOURCE_RUN_ID ?= dan-v2-20260106
OUT_RUN_ID ?= contractir-v0_2-strategy-tests-dev
EXP_FILTER ?=
CONTRACTIR_JSON ?=
FLOW_RUN_ID ?= contractir-v0_2-flow-dev
FLOW_SOURCE_RUN_ID ?= dan-v2-20260106
FLOW_ITEM_ID ?=

.PHONY: contractir-v0_2-strategy
contractir-v0_2-strategy:
	@FILTER_ARGS=""; \
	if [ -n "$(EXP_FILTER)" ]; then FILTER_ARGS="--exp-filter $(EXP_FILTER)"; fi; \
	$(POETRY) run pipeline contractir-v0-2 strategy-harness --source-run-id $(SOURCE_RUN_ID) --out-run-id $(OUT_RUN_ID) $$FILTER_ARGS

.PHONY: contractir-v0_2-validate
contractir-v0_2-validate:
	@if [ -z "$(CONTRACTIR_JSON)" ]; then echo "Set CONTRACTIR_JSON=path/to/contract_ir.json"; exit 2; fi
	$(POETRY) run pipeline contractir-v0-2 validate --path $(CONTRACTIR_JSON)

.PHONY: contractir-v0_2-flow
contractir-v0_2-flow:
	@ITEM_ARGS=""; \
	if [ -n "$(FLOW_ITEM_ID)" ]; then ITEM_ARGS="--item-id $(FLOW_ITEM_ID)"; fi; \
	$(POETRY) run pipeline contractir-v0-2 flow --run-id $(FLOW_RUN_ID) --source-run-id $(FLOW_SOURCE_RUN_ID) $$ITEM_ARGS
