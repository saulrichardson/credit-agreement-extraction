#!/usr/bin/env bash
set -euo pipefail

# Run QA instructions (explicit paths) in parallel or sequentially.
# Usage:
#   ./scripts/run_manual_agents_qc.sh -P 9 scratch/instructions/manual_agents_qc/Agent*.txt
#   ./scripts/run_manual_agents_qc.sh scratch/instructions/manual_agents_qc/Agent01.txt ...

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/logs/manual_agents"
FLAGS_STR="--json --skip-git-repo-check --full-auto --sandbox danger-full-access"
PARALLEL=0
mkdir -p "$LOG_DIR"

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

run_one() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing instruction file: $path" >&2
    return 1
  fi
  local base ts log
  base="$(basename "$path" .txt)"
  ts="$(date +%Y%m%dT%H%M%S)"
  log="$LOG_DIR/${base}_${ts}.jsonl"
  echo "Running codex exec for $(basename "$path") -> $log"
  codex exec $FLAGS_STR "$(cat "$path")" > "$log"
}

if [[ "$#" -eq 0 ]]; then
  echo "Provide instruction file paths (e.g., scratch/instructions/manual_agents_qc/Agent01.txt ...)" >&2
  exit 1
fi

if [[ "$PARALLEL" -gt 0 ]]; then
  printf '%s\0' "$@" | xargs -0 -P "$PARALLEL" -n1 bash -c '
    path="$1"
    LOG_DIR="'"$LOG_DIR"'"
    FLAGS_STR="'"$FLAGS_STR"'"
    if [[ ! -f "$path" ]]; then
      echo "Missing instruction file: $path" >&2
      exit 1
    fi
    base=$(basename "$path" .txt)
    ts="$(date +%Y%m%dT%H%M%S)"
    log="$LOG_DIR/${base}_${ts}.jsonl"
    echo "Running codex exec for $(basename "$path") -> $log"
    codex exec $FLAGS_STR "$(cat "$path")" > "$log"
  ' bash
else
  for path in "$@"; do
    run_one "$path"
  done
fi
