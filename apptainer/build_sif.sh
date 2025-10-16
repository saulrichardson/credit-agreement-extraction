#!/usr/bin/env bash
set -euo pipefail

DEF=${1:-edgar_filing.def}
SIF=${2:-edgar-filing.sif}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$SCRATCH/apptainer_tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$SCRATCH/apptainer_cache}"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true

if command -v apptainer >/dev/null 2>&1; then
    RUNTIME=apptainer
elif command -v singularity >/dev/null 2>&1; then
    RUNTIME=singularity
else
    echo "Neither apptainer nor singularity is available in PATH" >&2
    exit 1
fi

$RUNTIME build \
    --fakeroot \
    --bind "${REPO_ROOT}:/workspace/edgar-filing-pipeline" \
    "$SIF" "$(dirname "$0")/$DEF"

