#!/bin/bash

set -euo pipefail

GENERATOR="Unix Makefiles"

if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

TORCH_CMAKE_PREFIX=/opt/libtorch

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
