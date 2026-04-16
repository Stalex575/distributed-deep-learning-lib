# Distributed Deep Learning Lib

## Prerequisites

Install system dependencies on Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  python3-dev python3-venv \
  git cmake ninja-build \
  libopenmpi-dev openmpi-bin
```

If you use a custom MPI install (for example `$HOME/.local/mpich`), keep it available and set `MPI_HOME` when running setup scripts.

## Virtual Environment Setup From Scratch

Create and populate `.venv` with Python deps and a Torch wheel:

```bash
./scripts/setup_venv.sh
source .venv/bin/activate
```

Optional: install a specific CUDA wheel (example for cu130):

```bash
TORCH_SPEC="torch==2.10.0" \
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130" \
./scripts/setup_venv.sh
```

## Build Torch With c10d MPI Support (Required For C++ torch.distributed Example)

The `resnet_torch_dist_training` target requires Torch built from source with `USE_MPI=1`.

```bash
source .venv/bin/activate
MPI_HOME="$HOME/.local/mpich" \
PYTORCH_TAG="v2.10.0" \
USE_CUDA=0 \
./scripts/build_torch_with_mpi.sh
```

Notes:

- `USE_CUDA=0` is the fastest path to make CPU torch.distributed C++ run.
- Set `USE_CUDA=1` only if you intentionally want to build CUDA-enabled Torch from source and your CUDA toolchain is ready.
- Source builds are heavy and can take a long time.

## Build This Project

### Default build (custom MPI path)

```bash
cmake -B build -S .
cmake --build build -j
```

### Build with MPI allreduce in custom backend

```bash
cmake -B build -S . -DDISTDL_USE_MPI_ALLREDUCE=ON
cmake --build build -j
```

### Build including torch.distributed C++ example

```bash
cmake -B build -S . -DENABLE_TORCH_DIST_CPP_EXAMPLE=ON
cmake --build build -j
```

If Torch does not export `c10d::ProcessGroupMPI::createProcessGroupMPI`, CMake skips `resnet_torch_dist_training` with a warning.

## Run

### CPU only (8 MPI processes) custom implementation

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 mpirun -np 8 ./build/resnet_training
```

### CPU only (8 MPI processes) torch.distributed C++ example

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 mpirun -np 8 ./build/resnet_torch_dist_training
```

### One-command verify for both CPU examples

```bash
NP=8 ./scripts/verify_examples_cpu.sh
```

### CUDA with custom MPI path

```bash
mpirun -np 2 ./build/resnet_training
```

## Notes

- The custom training path is the default and uses `distributed_ops.h` and `distributed_training_utils.h`.
- `DISTDL_USE_MPI_ALLREDUCE` only changes allreduce implementation inside the custom MPI backend.
- `local_rank` is used for GPU selection when CUDA is visible.
