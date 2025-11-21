#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper to run all QA instructions (Agent01, Agent02, Agent03, Agent04,
# Agent05, Agent06, Agent07, Agent08, Agent10) in parallel.
# Usage:
#   ./scripts/run_manual_agents_qc_all.sh         # parallel with default concurrency (9)
#   ./scripts/run_manual_agents_qc_all.sh -P 4    # parallel with concurrency 4

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QC_DIR="$ROOT/scratch/instructions/manual_agents_qc"
LOG_DIR="$ROOT/logs/manual_agents"
FLAGS_STR="--json --skip-git-repo-check --full-auto --sandbox danger-full-access"
PARALLEL=9

# Parse optional -P/--parallel
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -P|--parallel)
      PARALLEL="$2"
      shift 2
      ;;
    --) shift; break ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

mkdir -p "$LOG_DIR"

files=(
  "$QC_DIR/Agent01.txt"
  "$QC_DIR/Agent02.txt"
  "$QC_DIR/Agent03.txt"
  "$QC_DIR/Agent04.txt"
  "$QC_DIR/Agent05.txt"
  "$QC_DIR/Agent06.txt"
  "$QC_DIR/Agent07.txt"
  "$QC_DIR/Agent08.txt"
  "$QC_DIR/Agent10.txt"
)

for f in "${files[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing instruction file: $f" >&2
    exit 1
  fi
done

printf '%s\0' "${files[@]}" | xargs -0 -P "$PARALLEL" -n1 bash -c '
  path="$1"
  LOG_DIR="'"$LOG_DIR"'"
  base=$(basename "$path" .txt)
  ts="$(date +%Y%m%dT%H%M%S)"
  log="$LOG_DIR/${base}_${ts}.jsonl"
  echo "Running codex exec for $(basename "$path") -> $log"
  FLAGS_STR="'"$FLAGS_STR"'"
  codex exec $FLAGS_STR "$(cat "$path")" > "$log"
' bash
