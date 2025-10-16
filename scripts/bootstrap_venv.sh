#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${1:-.venv}"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip
pip install -r "${ROOT_DIR}/requirements.txt"
pip install -e "${ROOT_DIR}"

echo "Virtual environment created at ${VENV_DIR}"
echo "Activate with: source ${VENV_DIR}/bin/activate"
