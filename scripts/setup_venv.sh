#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.10.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy pyyaml ninja typing_extensions sympy networkx jinja2 fsspec

if [[ -n "$TORCH_INDEX_URL" ]]; then
    python -m pip install "$TORCH_SPEC" --index-url "$TORCH_INDEX_URL"
else
    python -m pip install "$TORCH_SPEC"
fi

python - <<'PY'
import torch
print(f"Torch version: {torch.__version__}")
print(f"Torch CMake prefix: {torch.utils.cmake_prefix_path}")
PY

echo "Virtual environment ready at $VENV_DIR"
