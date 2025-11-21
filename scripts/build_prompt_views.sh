#!/usr/bin/env bash
set -euo pipefail

# Build prompt_view bundles for a list of accessions, given extracted EX-10 HTMLs.
# Input: file with one accession per line (e.g., accessions_new.txt).
# Expects HTML at scratch/extracted_ex10/<acc>_EX-10.html.
# Writes bundle to scratch/prompt_views/<acc>/ (disposable scratch; can regenerate).
# Skips any accession that already has scratch/prompt_views/<acc>/prompt_view.txt
# unless OVERWRITE=1.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HTML_ROOT="$ROOT/scratch/extracted_ex10"
OUT_ROOT="$ROOT/scratch/prompt_views"
ACCESSIONS_FILE="${1:-}"

if [[ -z "$ACCESSIONS_FILE" || ! -f "$ACCESSIONS_FILE" ]]; then
  echo "Usage: $0 accessions.txt" >&2
  exit 1
fi

OVERWRITE=${OVERWRITE:-0}

while IFS= read -r acc; do
  [[ -z "$acc" ]] && continue
  html="$HTML_ROOT/${acc}_EX-10.html"
  outdir="$OUT_ROOT/$acc"
  if [[ ! -f "$html" ]]; then
    echo "[skip] missing HTML: $html" >&2
    continue
  fi
  if [[ -f "$outdir/prompt_view.txt" && "$OVERWRITE" != "1" ]]; then
    echo "[skip] prompt_view exists: $outdir/prompt_view.txt" >&2
    continue
  fi
  rm -rf "$outdir"
  # anchor-document refuses to overwrite an existing dir; give it a fresh temp name
  tmpdir=$(mktemp -u)
  poetry run edgar-pipeline anchor-document --input-html "$html" --output-dir "$tmpdir"
  mkdir -p "$outdir"
  mv "$tmpdir"/* "$outdir"/
  rmdir "$tmpdir" || true
  echo "[ok] built $outdir/prompt_view.txt"
done < "$ACCESSIONS_FILE"
