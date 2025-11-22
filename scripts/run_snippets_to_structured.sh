#!/usr/bin/env bash
set -euo pipefail

# Second-pass structured extraction over the all-in-one snippets using the
# fundamentals+pricing JSON prompt at prompts/dg_v3.txt.
#
# Inputs: runs/<run_id>/snippets/*_snippets.txt (from run_all_in_one_pipeline.sh)
# Output: runs/<run_id>/json_structured/<accession>.txt (raw JSON per accession)
#
# Usage:
#   ./scripts/run_snippets_to_structured.sh [run_id]
# Optional:
#   WORKERS (default 6)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKERS="${WORKERS:-6}"
run_id="${1:-${RUN_ID:-}}"
PROMPT="$ROOT/scratch/prompts/dg_v3.txt"

if [[ -z "$run_id" ]]; then
  # pick most recent directory under runs/ (directories only, by mtime)
  run_id="$(ls -1t "$ROOT/runs" | while read -r f; do
    [[ -d "$ROOT/runs/$f" ]] && echo "$f" && break
  done)"
fi

SNIP_DIR="$ROOT/runs/$run_id/snippets"
OUT_DIR="$ROOT/runs/$run_id/json_structured"

if [[ ! -d "$SNIP_DIR" ]]; then
  echo "Missing snippets dir: $SNIP_DIR. Run run_all_in_one_pipeline.sh first." >&2
  exit 1
fi
if [[ ! -f "$PROMPT" ]]; then
  echo "Missing prompt: $PROMPT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

python "$ROOT/scripts/run_reasoning_batch.py" \
  --input-dir "$SNIP_DIR" \
  --pattern "*_snippets.txt" \
  --prompt-file "$PROMPT" \
  --output-dir "$OUT_DIR" \
  --workers "$WORKERS"

echo "Structured JSON written to $OUT_DIR"
