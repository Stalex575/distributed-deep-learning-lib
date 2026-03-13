#!/bin/bash

GENERATOR="Unix Makefiles"

if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

cd build

echo "Generating CMake files..."

cmake -G "$GENERATOR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=/opt/libtorch \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc \
      ..

if [ $? -ne 0 ]; then
    echo "[ERROR] CMake generation failed!"
    exit 1
fi

echo "Compiling project..."
cmake --build . --config Release --parallel

if [ $? -ne 0 ]; then
    echo "[ERROR] Compilation failed!"
    exit 1
fi

echo ""
echo "Build complete!"
echo ""
