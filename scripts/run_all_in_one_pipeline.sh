#!/usr/bin/env bash
set -euo pipefail

# End-to-end all-in-one classification + snippet rendering for a list of accessions.
# - Requires prompt_view.txt at scratch/prompt_views/<acc>/prompt_view.txt
# - Uses prompt_all_comprehensive_v2.txt (already patched to {{document}})
# - Outputs run artifacts under runs/<run_id>/
#
# Usage:
#   ./scripts/run_all_in_one_pipeline.sh accessions.txt [run_id]
# where accessions.txt contains one accession per line (matching the prompt_view dirs).
# If run_id is omitted, defaults to RUN_ID env var or run_YYYYMMDD.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANCHORS_DIR="$ROOT/scratch/prompt_views"
PROMPT_FILE="$ROOT/prompts/prompt_all_comprehensive_v2.txt"
WORKERS="${WORKERS:-6}"
BANDWIDTH="${BANDWIDTH:-4}"

accessions_file="${1:-}"
run_id="${2:-${RUN_ID:-run_$(date +%Y%m%d)}}"

if [[ -z "$accessions_file" || ! -f "$accessions_file" ]]; then
  echo "Provide an accessions file (one accession per line)." >&2
  exit 1
fi

OUT_DIR="$ROOT/runs/$run_id"
CLS_DIR="$OUT_DIR/classifications"
SNIP_DIR="$OUT_DIR/snippets"
INPUT_DIR="$OUT_DIR/input_prompt_views"
mkdir -p "$CLS_DIR" "$SNIP_DIR" "$INPUT_DIR" "$OUT_DIR/logs"

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "Missing prompt file: $PROMPT_FILE" >&2
  exit 1
fi

accs=()
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  accs+=("$line")
done < <(grep -v '^[[:space:]]*$' "$accessions_file")

# Build a run-scoped input dir of copied prompt_views for reproducibility
rm -rf "$INPUT_DIR"
mkdir -p "$INPUT_DIR"
for acc in "${accs[@]}"; do
  pv="$ANCHORS_DIR/$acc/prompt_view.txt"
  if [[ ! -f "$pv" ]]; then
    echo "Missing prompt_view: $pv" >&2
    exit 1
  fi
  cp "$pv" "$INPUT_DIR/${acc}.txt"
done

# Run classification
python "$ROOT/scripts/run_reasoning_batch.py" \
  --input-dir "$INPUT_DIR" \
  --pattern "*.txt" \
  --prompt-file "$PROMPT_FILE" \
  --output-dir "$CLS_DIR" \
  --workers "$WORKERS"

# Render snippets
for cls in "$CLS_DIR"/*.txt; do
  acc=$(basename "${cls%.txt}")
  pv="$ANCHORS_DIR/$acc/prompt_view.txt"
  python "$ROOT/scripts/render_anchor_snippets.py" \
    --prompt-view "$pv" \
    --classification "$cls" \
    --bandwidth "$BANDWIDTH" \
    --include-headers \
    --output "$SNIP_DIR/${acc}_snippets.txt"
done

echo "Done. Run: $run_id"
echo "  Input prompt_views -> $INPUT_DIR"
echo "  Classifications   -> $CLS_DIR"
echo "  Snippets          -> $SNIP_DIR (bandwidth=$BANDWIDTH)"
