#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
NP="${NP:-8}"

# shellcheck disable=SC1090
source "$ROOT_DIR/.venv/bin/activate"

cmake -B "$BUILD_DIR" -S "$ROOT_DIR" -DENABLE_TORCH_DIST_CPP_EXAMPLE=ON
cmake --build "$BUILD_DIR" -j

if [[ ! -x "$BUILD_DIR/resnet_torch_dist_training" ]]; then
    echo "resnet_torch_dist_training was not built."
    echo "Your Torch build likely does not export ProcessGroupMPI."
    exit 1
fi

CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 mpirun -np "$NP" "$BUILD_DIR/resnet_training"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 mpirun -np "$NP" "$BUILD_DIR/resnet_torch_dist_training"
