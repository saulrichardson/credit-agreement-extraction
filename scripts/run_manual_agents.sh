#!/usr/bin/env bash
set -euo pipefail

# Run codex exec for one or more agent instruction files.
# Usage:
#   ./scripts/run_manual_agents.sh                  # run all Agent*.txt sequentially
#   ./scripts/run_manual_agents.sh -P 8             # run all in parallel (8 at a time)
#   ./scripts/run_manual_agents.sh Agent12          # run just Agent12 sequentially
#   ./scripts/run_manual_agents.sh -P 4 Agent01 Agent05 Agent09   # run selected in parallel
#
# Expects instruction files at instructions/manual_agents/AgentXX.txt.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTR_DIR="$ROOT/instructions/manual_agents"
LOG_DIR="$ROOT/logs/manual_agents"
FLAGS=(--json --skip-git-repo-check --full-auto --sandbox danger-full-access)
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

run_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing instruction file: $path" >&2
    return 1
  fi
  local base
  base="$(basename "$path" .txt)"
  local log="$LOG_DIR/${base}.jsonl"
  echo "Running codex exec for $(basename "$path") -> $log"
  # Capture full JSON event stream; Codex writes events to stdout in JSONL.
  codex exec "${FLAGS[@]}" "$(cat "$path")" | tee "$log"
}

if [[ "$#" -eq 0 ]]; then
  shopt -s nullglob
  files=("$INSTR_DIR"/Agent*.txt)
  if [[ "${#files[@]}" -eq 0 ]]; then
    echo "No instruction files found in $INSTR_DIR" >&2
    exit 1
  fi
  if [[ "$PARALLEL" -gt 0 ]]; then
    printf '%s\0' "${files[@]}" | xargs -0 -P "$PARALLEL" -n1 bash -c '
      path="$1"
      ROOT="'"$ROOT"'"
      LOG_DIR="'"$LOG_DIR"'"
      FLAGS_STR="'"$FLAGS_STR"'"
      base=$(basename "$path" .txt)
      log="$LOG_DIR/${base}.jsonl"
      echo "Running codex exec for $(basename "$path") -> $log"
      codex exec $FLAGS_STR "$(cat "$path")" | tee "$log"
    ' bash
  else
    for file in "${files[@]}"; do
      run_file "$file"
    done
  fi
else
  # remaining args are agent names
  files=()
  for agent in "$@"; do
    base="${agent%.txt}"
    files+=("$INSTR_DIR/${base}.txt")
  done
  if [[ "$PARALLEL" -gt 0 ]]; then
    printf '%s\0' "${files[@]}" | xargs -0 -P "$PARALLEL" -n1 bash -c '
      path="$1"
      ROOT="'"$ROOT"'"
      LOG_DIR="'"$LOG_DIR"'"
      FLAGS_STR="'"$FLAGS_STR"'"
      if [[ ! -f "$path" ]]; then
        echo "Missing instruction file: $path" >&2
        exit 1
      fi
      base=$(basename "$path" .txt)
      log="$LOG_DIR/${base}.jsonl"
      echo "Running codex exec for $(basename "$path") -> $log"
      codex exec $FLAGS_STR "$(cat "$path")" | tee "$log"
    ' bash
  else
    for path in "${files[@]}"; do
      run_file "$path"
    done
  fi
fi
