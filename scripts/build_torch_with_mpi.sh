#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTORCH_SRC_DIR="${PYTORCH_SRC_DIR:-$ROOT_DIR/third_party/pytorch}"
PYTORCH_TAG="${PYTORCH_TAG:-v2.10.0}"
MPI_HOME="${MPI_HOME:-$HOME/.local/mpich}"
USE_CUDA="${USE_CUDA:-0}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Missing virtual environment: $VENV_DIR"
    echo "Run scripts/setup_venv.sh first."
    exit 1
fi

if [[ ! -x "$MPI_HOME/bin/mpirun" ]]; then
    echo "MPI runtime not found at $MPI_HOME/bin/mpirun"
    echo "Set MPI_HOME to your MPI install root."
    exit 1
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install cmake ninja pyyaml typing_extensions sympy networkx jinja2 fsspec six

mkdir -p "$(dirname "$PYTORCH_SRC_DIR")"
if [[ ! -d "$PYTORCH_SRC_DIR/.git" ]]; then
    git clone --recursive https://github.com/pytorch/pytorch "$PYTORCH_SRC_DIR"
fi

pushd "$PYTORCH_SRC_DIR" >/dev/null

git fetch --tags --force origin

git checkout "$PYTORCH_TAG"
git submodule sync --recursive
git submodule update --init --recursive

export PATH="$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"

export USE_DISTRIBUTED=1
export USE_MPI=1
export USE_GLOO=0
export USE_CUDA="$USE_CUDA"
export BUILD_TEST=0
export MAX_JOBS="$MAX_JOBS"

python -m pip uninstall -y torch torchvision torchaudio || true
python setup.py bdist_wheel
python -m pip install --force-reinstall dist/torch-*.whl

popd >/dev/null

python - <<'PY'
import torch
print(f"Torch version: {torch.__version__}")
print(f"Torch CMake prefix: {torch.utils.cmake_prefix_path}")
PY

echo "Built and installed Torch from source with USE_MPI=1"
