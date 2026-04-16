#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
GENERATOR="${GENERATOR:-Unix Makefiles}"
JOBS="${JOBS:-$(nproc)}"

echo "Generating CMake files..."
cmake -G "$GENERATOR" \
    -B "$BUILD_DIR" \
    -S "$ROOT_DIR" \
    "$@"

echo "Compiling project..."
cmake --build "$BUILD_DIR" --parallel "$JOBS"

echo ""
echo "Build complete!"
echo ""
