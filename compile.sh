#!/bin/bash

set -euo pipefail

GENERATOR="Unix Makefiles"
PYTHON_BIN="$(pwd)/.venv/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "[ERROR] Python not found at $PYTHON_BIN"
    echo "Create a venv and install torch first:"
    echo "  python3 -m venv .venv"
    echo "  .venv/bin/python -m pip install torch torchvision"
    exit 1
fi

TORCH_CMAKE_PREFIX="$($PYTHON_BIN -c 'import torch; print(torch.utils.cmake_prefix_path)')"

if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

cd build

echo "Generating CMake files..."

cmake -G "$GENERATOR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX" \
      ..

echo "Compiling project..."
cmake --build . --config Release --parallel

echo ""
echo "Build complete!"
echo ""
