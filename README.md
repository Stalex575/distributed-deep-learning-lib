# Distributed Deep Learning Lib

## Build

The project uses the Python venv Torch install in this workspace.

### Default build

This builds the custom MPI implementation with the default `reduce + broadcast` allreduce path.

```bash
cmake -B build -S .
cmake --build build -j
```

### Build with MPI allreduce

This enables the direct MPI allreduce path via `DISTDL_USE_MPI_ALLREDUCE`.

```bash
cmake -B build -S . -DDISTDL_USE_MPI_ALLREDUCE=ON
cmake --build build -j
```

### Optional torch.distributed C++ example

```bash
cmake -B build -S . -DENABLE_TORCH_DIST_CPP_EXAMPLE=ON
cmake --build build -j
```

## Run

### CPU only

Hide GPUs so the custom training binary stays on CPU.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 mpirun -np 2 ./build/resnet_training
```

### CUDA with custom MPI path

Leave GPUs visible and launch one MPI process per GPU.

```bash
mpirun -np 2 ./build/resnet_training
```

### CUDA with MPI allreduce

Build first with `-DDISTDL_USE_MPI_ALLREDUCE=ON`, then run the same way.

```bash
mpirun -np 2 ./build/resnet_training
```

### Optional torch.distributed C++ example

If enabled in CMake, the target is `resnet_torch_dist_training`.

```bash
CUDA_VISIBLE_DEVICES="" mpirun -np 2 ./build/resnet_torch_dist_training
```

## Notes

- The custom training path is the default and uses the code in `distributed_ops.h` and `distributed_training_utils.h`.
- `DISTDL_USE_MPI_ALLREDUCE` only changes the allreduce implementation inside the MPI backend.
- `local_rank` is used for GPU selection when CUDA is visible.