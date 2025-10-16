#!/usr/bin/env bash
set -euo pipefail

SIZE_MB=${SIZE_MB:-4096}
SIF_NAME=${SIF_NAME:-python311.sif}
OVERLAY_NAME=${OVERLAY_NAME:-edgar-filing.ext3}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SCRATCH_DIR=${SCRATCH:-/scratch/$USER}
TARGET_DIR=${1:-$SCRATCH_DIR/repos/edgar-filing-pipeline}
mkdir -p "$(dirname "$TARGET_DIR")"
if [ ! -d "$TARGET_DIR" ]; then
    git clone git@github.com:saulrichardson/edgar-filing-pipeline.git "$TARGET_DIR"
fi

cd "$TARGET_DIR"

if ! command -v singularity >/dev/null 2>&1; then
    echo "singularity command not found. Please load the module first." >&2
    exit 1
fi

if [ ! -f "$SIF_NAME" ]; then
    echo "Pulling base image $SIF_NAME ..."
    singularity pull "$SIF_NAME" docker://python:3.11-slim
fi

if [ ! -f "$OVERLAY_NAME" ]; then
    echo "Creating writable overlay $OVERLAY_NAME (${SIZE_MB} MB) ..."
    singularity overlay create --size "$SIZE_MB" "$OVERLAY_NAME"
fi

echo "Bootstrapping virtualenv inside overlay ..."
singularity exec --overlay "$OVERLAY_NAME":rw "$TARGET_DIR/$SIF_NAME" python -m venv /ext3/venv

echo "Installing pinned requirements ..."
singularity exec --overlay "$OVERLAY_NAME":rw --bind "$TARGET_DIR:/workspace" "$TARGET_DIR/$SIF_NAME" \
    /ext3/venv/bin/pip install --no-cache-dir -r /workspace/requirements.txt

echo "Installing edgar-filing-pipeline package ..."
singularity exec --overlay "$OVERLAY_NAME":rw --bind "$TARGET_DIR:/workspace" "$TARGET_DIR/$SIF_NAME" \
    /ext3/venv/bin/pip install --no-cache-dir /workspace

echo
cat <<MSG
Overlay environment ready.
Invoke it with:
  REPO=$TARGET_DIR
  singularity exec --overlay \$REPO/$OVERLAY_NAME \
    --bind \$REPO:/workspace \$REPO/$SIF_NAME \
    /ext3/venv/bin/python -m edgar_filing_pipeline.scripts.build_manifest ...
MSG
