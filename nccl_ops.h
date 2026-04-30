#pragma once

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#if defined(DISTDL_HAS_NCCL) && DISTDL_HAS_NCCL

#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <nccl.h>

#if defined(DISTDL_HAS_MPI) && DISTDL_HAS_MPI
#include <mpi.h>
#endif

namespace distdl::distributed::nccl_backend
{
enum class ReduceOp { Sum, Product, Min, Max };

namespace detail
{
struct State
{
    ncclComm_t comm = nullptr;
    int rank = 0;
    int world = 1;
    int local_rank = 0;
    int device_index = 0;
    bool initialized = false;
    bool mpi_initialized_by_us = false;
};

inline State& state()
{
    static State s;
    return s;
}

inline void nccl_check(ncclResult_t r, const char* where)
{
    if (r != ncclSuccess)
        throw std::runtime_error(std::string("NCCL error in ") + where + ": " + ncclGetErrorString(r));
}

inline void cuda_check(cudaError_t r, const char* where)
{
    if (r != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error in ") + where + ": " + cudaGetErrorString(r));
}

inline ncclDataType_t to_nccl_dtype(torch::ScalarType dt)
{
    switch (dt)
    {
    case torch::kFloat32:  return ncclFloat32;
    case torch::kFloat64:  return ncclFloat64;
    case torch::kFloat16:  return ncclFloat16;
    case torch::kBFloat16: return ncclBfloat16;
    case torch::kInt32:    return ncclInt32;
    case torch::kInt64:    return ncclInt64;
    case torch::kUInt8:    return ncclUint8;
    case torch::kInt8:     return ncclInt8;
    default:
        throw std::runtime_error(std::string("NCCL: unsupported dtype ") + toString(dt));
    }
}

inline ncclRedOp_t to_nccl_op(ReduceOp op)
{
    switch (op)
    {
    case ReduceOp::Sum:     return ncclSum;
    case ReduceOp::Product: return ncclProd;
    case ReduceOp::Min:     return ncclMin;
    case ReduceOp::Max:     return ncclMax;
    }
    throw std::runtime_error("NCCL: bad reduce op");
}

inline int try_env_int(const char* name)
{
    const char* v = std::getenv(name);
    if (!v || !v[0]) return -1;
    try { return std::stoi(v); } catch (...) { return -1; }
}

inline int detect_local_rank(int world_rank_fallback)
{
    for (const char* k : {"OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK",
                           "MPI_LOCALRANKID", "SLURM_LOCALID"})
    {
        int r = try_env_int(k);
        if (r >= 0) return r;
    }
    return world_rank_fallback;
}

inline torch::Device active_device()
{
    return torch::Device(torch::kCUDA, state().device_index);
}

inline torch::Tensor stage_to_device(const torch::Tensor& t)
{
    if (!t.defined()) throw std::runtime_error("NCCL op received undefined tensor");
    auto dev = active_device();
    torch::Tensor out = t;
    if (!out.is_cuda() || out.device().index() != state().device_index)
        out = out.to(dev);
    return out.contiguous();
}

inline void copy_back(const torch::Tensor& staged, torch::Tensor& dst)
{
    if (!dst.defined())
    {
        dst = staged;
        return;
    }
    if (dst.is_cuda() && dst.device().index() == state().device_index && dst.is_contiguous()
        && dst.scalar_type() == staged.scalar_type() && dst.sizes() == staged.sizes())
    {
        if (dst.data_ptr() != staged.data_ptr()) dst.copy_(staged);
    }
    else
    {
        dst.copy_(staged.to(dst.device(), dst.scalar_type()));
    }
}

inline cudaStream_t current_stream()
{
    return c10::cuda::getCurrentCUDAStream(state().device_index).stream();
}

inline void stream_sync()
{
    cuda_check(cudaStreamSynchronize(current_stream()), "cudaStreamSynchronize");
}
}

inline void init()
{
    if (detail::state().initialized) return;

#if !defined(DISTDL_HAS_MPI) || !(DISTDL_HAS_MPI)
    throw std::runtime_error("NCCL bootstrap requires MPI for unique-id broadcast at startup");
#else
    int mpi_init_flag = 0;
    if (MPI_Initialized(&mpi_init_flag) != MPI_SUCCESS)
        throw std::runtime_error("NCCL init: MPI_Initialized failed");

    if (!mpi_init_flag)
    {
        int argc = 0; char** argv = nullptr; int provided = 0;
        if (MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided) != MPI_SUCCESS)
            throw std::runtime_error("NCCL init: MPI_Init_thread failed");
        detail::state().mpi_initialized_by_us = true;
    }

    int mpi_rank = 0, mpi_world = 1;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) != MPI_SUCCESS ||
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_world) != MPI_SUCCESS)
        throw std::runtime_error("NCCL init: failed to query MPI rank/world");

    int gpu_count = 0;
    detail::cuda_check(cudaGetDeviceCount(&gpu_count), "cudaGetDeviceCount");
    if (gpu_count <= 0)
        throw std::runtime_error("NCCL init: no CUDA devices visible");

    const int local = detail::detect_local_rank(mpi_rank);
    const int dev = local % gpu_count;
    detail::cuda_check(cudaSetDevice(dev), "cudaSetDevice");

    ncclUniqueId id{};
    if (mpi_rank == 0)
        detail::nccl_check(ncclGetUniqueId(&id), "ncclGetUniqueId");

    if (MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD) != MPI_SUCCESS)
        throw std::runtime_error("NCCL init: MPI_Bcast(uniqueId) failed");

    ncclComm_t comm = nullptr;
    detail::nccl_check(ncclCommInitRank(&comm, mpi_world, id, mpi_rank), "ncclCommInitRank");

    detail::state().comm = comm;
    detail::state().rank = mpi_rank;
    detail::state().world = mpi_world;
    detail::state().local_rank = local;
    detail::state().device_index = dev;
    detail::state().initialized = true;
#endif
}

inline void finalize()
{
    auto& s = detail::state();
    if (s.initialized)
    {
        ncclCommDestroy(s.comm);
        s.comm = nullptr;
        s.initialized = false;
    }

#if defined(DISTDL_HAS_MPI) && DISTDL_HAS_MPI
    if (s.mpi_initialized_by_us)
    {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) MPI_Finalize();
        s.mpi_initialized_by_us = false;
    }
#endif
}

inline bool is_initialized() { return detail::state().initialized; }
inline int get_rank()        { return detail::state().rank; }
inline int get_world_size()  { return detail::state().world; }
inline int get_local_rank()  { return detail::state().local_rank; }

inline void barrier()
{
    if (!is_initialized()) throw std::runtime_error("NCCL not initialized");
    auto& s = detail::state();
    c10::cuda::CUDAGuard guard(s.device_index);
    auto stream = detail::current_stream();
    auto sync_buf = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(detail::active_device()));
    detail::nccl_check(
        ncclAllReduce(sync_buf.data_ptr(), sync_buf.data_ptr(), 1,
                      ncclFloat32, ncclSum, s.comm, stream),
        "ncclAllReduce(barrier)");
    detail::stream_sync();
}

inline void broadcast(torch::Tensor& tensor, int src)
{
    if (!is_initialized()) throw std::runtime_error("NCCL not initialized");
    auto& s = detail::state();
    c10::cuda::CUDAGuard guard(s.device_index);

    torch::Tensor staged = detail::stage_to_device(tensor);
    detail::nccl_check(
        ncclBcast(staged.data_ptr(), static_cast<size_t>(staged.numel()),
                  detail::to_nccl_dtype(staged.scalar_type()), src, s.comm,
                  detail::current_stream()),
        "ncclBcast");
    detail::stream_sync();
    detail::copy_back(staged, tensor);
}

inline void allreduce(torch::Tensor& tensor, ReduceOp op = ReduceOp::Sum)
{
    if (!is_initialized()) throw std::runtime_error("NCCL not initialized");
    auto& s = detail::state();
    c10::cuda::CUDAGuard guard(s.device_index);

    torch::Tensor staged = detail::stage_to_device(tensor);
    detail::nccl_check(
        ncclAllReduce(staged.data_ptr(), staged.data_ptr(),
                      static_cast<size_t>(staged.numel()),
                      detail::to_nccl_dtype(staged.scalar_type()),
                      detail::to_nccl_op(op), s.comm, detail::current_stream()),
        "ncclAllReduce");
    detail::stream_sync();
    detail::copy_back(staged, tensor);
}

inline void reduce(const torch::Tensor& input, torch::Tensor& output, int dst, ReduceOp op = ReduceOp::Sum)
{
    if (!is_initialized()) throw std::runtime_error("NCCL not initialized");
    if (input.sizes() != output.sizes() || input.scalar_type() != output.scalar_type())
        throw std::runtime_error("NCCL reduce: shape/dtype mismatch");

    auto& s = detail::state();
    c10::cuda::CUDAGuard guard(s.device_index);

    torch::Tensor send_buf = detail::stage_to_device(input);
    torch::Tensor recv_buf = (s.rank == dst)
        ? detail::stage_to_device(output)
        : torch::empty_like(send_buf);

    detail::nccl_check(
        ncclReduce(send_buf.data_ptr(), recv_buf.data_ptr(),
                   static_cast<size_t>(send_buf.numel()),
                   detail::to_nccl_dtype(send_buf.scalar_type()),
                   detail::to_nccl_op(op), dst, s.comm, detail::current_stream()),
        "ncclReduce");
    detail::stream_sync();

    if (s.rank == dst) detail::copy_back(recv_buf, output);
}

inline void send(const torch::Tensor& tensor, int dst, int /*tag*/ = 0)
{
    if (!is_initialized()) throw std::runtime_error("NCCL not initialized");
    auto& s = detail::state();
    c10::cuda::CUDAGuard guard(s.device_index);

    torch::Tensor staged = detail::stage_to_device(tensor);
    auto stream = detail::current_stream();

    int64_t header[34] = {0};
    header[0] = static_cast<int64_t>(staged.scalar_type());
    header[1] = staged.dim();
    for (int64_t i = 0; i < staged.dim(); ++i) header[2 + i] = staged.size(i);

    auto header_t = torch::from_blob(header, {34}, torch::TensorOptions().dtype(torch::kInt64))
                        .to(detail::active_device());

    detail::nccl_check(ncclGroupStart(), "ncclGroupStart(send)");
    detail::nccl_check(
        ncclSend(header_t.data_ptr(), 34, ncclInt64, dst, s.comm, stream),
        "ncclSend(header)");
    detail::nccl_check(
        ncclSend(staged.data_ptr(), static_cast<size_t>(staged.numel()),
                 detail::to_nccl_dtype(staged.scalar_type()), dst, s.comm, stream),
        "ncclSend(data)");
    detail::nccl_check(ncclGroupEnd(), "ncclGroupEnd(send)");
    detail::stream_sync();
}

inline void recv(torch::Tensor& tensor, int src, int /*tag*/ = 0)
{
    if (!is_initialized()) throw std::runtime_error("NCCL not initialized");
    auto& s = detail::state();
    c10::cuda::CUDAGuard guard(s.device_index);
    auto stream = detail::current_stream();

    auto header_t = torch::zeros({34}, torch::TensorOptions().dtype(torch::kInt64).device(detail::active_device()));
    detail::nccl_check(
        ncclRecv(header_t.data_ptr(), 34, ncclInt64, src, s.comm, stream),
        "ncclRecv(header)");
    detail::stream_sync();

    auto header_cpu = header_t.to(torch::kCPU);
    auto* hp = header_cpu.data_ptr<int64_t>();
    auto dt = static_cast<torch::ScalarType>(hp[0]);
    int64_t ndim = hp[1];
    if (ndim < 0 || ndim > 32) throw std::runtime_error("NCCL recv: bad ndim");

    std::vector<int64_t> shape(ndim);
    for (int64_t i = 0; i < ndim; ++i) shape[i] = hp[2 + i];

    auto staged = torch::empty(shape, torch::TensorOptions().dtype(dt).device(detail::active_device()));
    detail::nccl_check(
        ncclRecv(staged.data_ptr(), static_cast<size_t>(staged.numel()),
                 detail::to_nccl_dtype(dt), src, s.comm, stream),
        "ncclRecv(data)");
    detail::stream_sync();

    detail::copy_back(staged, tensor);
}

inline void gather(const torch::Tensor& input, std::vector<torch::Tensor>& gather_list, int dst)
{
    if (!is_initialized()) throw std::runtime_error("NCCL not initialized");
    auto& s = detail::state();
    c10::cuda::CUDAGuard guard(s.device_index);
    auto stream = detail::current_stream();

    if (s.rank == dst && static_cast<int>(gather_list.size()) != s.world)
        throw std::runtime_error("NCCL gather: list size != world");

    torch::Tensor staged_in = detail::stage_to_device(input);
    int64_t my_header[34] = {0};
    my_header[0] = static_cast<int64_t>(staged_in.scalar_type());
    my_header[1] = staged_in.dim();
    for (int64_t i = 0; i < staged_in.dim(); ++i) my_header[2 + i] = staged_in.size(i);

    if (s.rank == dst)
    {
        std::vector<torch::Tensor> headers(s.world);
        for (int r = 0; r < s.world; ++r)
            headers[r] = torch::zeros({34}, torch::TensorOptions().dtype(torch::kInt64).device(detail::active_device()));

        auto my_header_t = torch::from_blob(my_header, {34}, torch::TensorOptions().dtype(torch::kInt64))
                               .to(detail::active_device());
        headers[dst].copy_(my_header_t);

        detail::nccl_check(ncclGroupStart(), "ncclGroupStart(gather-headers)");
        for (int r = 0; r < s.world; ++r)
        {
            if (r == dst) continue;
            detail::nccl_check(
                ncclRecv(headers[r].data_ptr(), 34, ncclInt64, r, s.comm, stream),
                "ncclRecv(gather-header)");
        }
        detail::nccl_check(ncclGroupEnd(), "ncclGroupEnd(gather-headers)");
        detail::stream_sync();

        std::vector<torch::Tensor> staged_outs(s.world);
        staged_outs[dst] = staged_in;

        detail::nccl_check(ncclGroupStart(), "ncclGroupStart(gather-data)");
        for (int r = 0; r < s.world; ++r)
        {
            if (r == dst) continue;
            auto h_cpu = headers[r].to(torch::kCPU);
            auto* hp = h_cpu.data_ptr<int64_t>();
            auto dt = static_cast<torch::ScalarType>(hp[0]);
            int64_t ndim = hp[1];
            std::vector<int64_t> shape(ndim);
            for (int64_t i = 0; i < ndim; ++i) shape[i] = hp[2 + i];
            staged_outs[r] = torch::empty(shape, torch::TensorOptions().dtype(dt).device(detail::active_device()));
            detail::nccl_check(
                ncclRecv(staged_outs[r].data_ptr(), static_cast<size_t>(staged_outs[r].numel()),
                         detail::to_nccl_dtype(dt), r, s.comm, stream),
                "ncclRecv(gather-data)");
        }
        detail::nccl_check(ncclGroupEnd(), "ncclGroupEnd(gather-data)");
        detail::stream_sync();

        for (int r = 0; r < s.world; ++r)
            detail::copy_back(staged_outs[r], gather_list[r]);
    }
    else
    {
        auto my_header_t = torch::from_blob(my_header, {34}, torch::TensorOptions().dtype(torch::kInt64))
                               .to(detail::active_device());

        detail::nccl_check(ncclGroupStart(), "ncclGroupStart(gather-send-header)");
        detail::nccl_check(
            ncclSend(my_header_t.data_ptr(), 34, ncclInt64, dst, s.comm, stream),
            "ncclSend(gather-header)");
        detail::nccl_check(ncclGroupEnd(), "ncclGroupEnd(gather-send-header)");
        detail::stream_sync();

        detail::nccl_check(ncclGroupStart(), "ncclGroupStart(gather-send-data)");
        detail::nccl_check(
            ncclSend(staged_in.data_ptr(), static_cast<size_t>(staged_in.numel()),
                     detail::to_nccl_dtype(staged_in.scalar_type()), dst, s.comm, stream),
            "ncclSend(gather-data)");
        detail::nccl_check(ncclGroupEnd(), "ncclGroupEnd(gather-send-data)");
        detail::stream_sync();
    }
}

inline void scatter(torch::Tensor& output, const std::vector<torch::Tensor>& scatter_list, int src)
{
    if (!is_initialized()) throw std::runtime_error("NCCL not initialized");
    auto& s = detail::state();
    c10::cuda::CUDAGuard guard(s.device_index);
    auto stream = detail::current_stream();

    if (s.rank == src && static_cast<int>(scatter_list.size()) != s.world)
        throw std::runtime_error("NCCL scatter: list size != world");

    if (s.rank == src)
    {
        std::vector<torch::Tensor> staged(s.world);
        for (int r = 0; r < s.world; ++r) staged[r] = detail::stage_to_device(scatter_list[r]);

        detail::nccl_check(ncclGroupStart(), "ncclGroupStart(scatter-headers)");
        for (int r = 0; r < s.world; ++r)
        {
            if (r == src) continue;
            int64_t hdr[34] = {0};
            hdr[0] = static_cast<int64_t>(staged[r].scalar_type());
            hdr[1] = staged[r].dim();
            for (int64_t i = 0; i < staged[r].dim(); ++i) hdr[2 + i] = staged[r].size(i);
            auto hdr_t = torch::from_blob(hdr, {34}, torch::TensorOptions().dtype(torch::kInt64))
                             .clone().to(detail::active_device());
            detail::nccl_check(
                ncclSend(hdr_t.data_ptr(), 34, ncclInt64, r, s.comm, stream),
                "ncclSend(scatter-header)");
        }
        detail::nccl_check(ncclGroupEnd(), "ncclGroupEnd(scatter-headers)");
        detail::stream_sync();

        detail::nccl_check(ncclGroupStart(), "ncclGroupStart(scatter-data)");
        for (int r = 0; r < s.world; ++r)
        {
            if (r == src) continue;
            detail::nccl_check(
                ncclSend(staged[r].data_ptr(), static_cast<size_t>(staged[r].numel()),
                         detail::to_nccl_dtype(staged[r].scalar_type()), r, s.comm, stream),
                "ncclSend(scatter-data)");
        }
        detail::nccl_check(ncclGroupEnd(), "ncclGroupEnd(scatter-data)");
        detail::stream_sync();

        detail::copy_back(staged[src], output);
    }
    else
    {
        auto header_t = torch::zeros({34}, torch::TensorOptions().dtype(torch::kInt64).device(detail::active_device()));
        detail::nccl_check(ncclGroupStart(), "ncclGroupStart(scatter-recv-header)");
        detail::nccl_check(
            ncclRecv(header_t.data_ptr(), 34, ncclInt64, src, s.comm, stream),
            "ncclRecv(scatter-header)");
        detail::nccl_check(ncclGroupEnd(), "ncclGroupEnd(scatter-recv-header)");
        detail::stream_sync();

        auto h_cpu = header_t.to(torch::kCPU);
        auto* hp = h_cpu.data_ptr<int64_t>();
        auto dt = static_cast<torch::ScalarType>(hp[0]);
        int64_t ndim = hp[1];
        std::vector<int64_t> shape(ndim);
        for (int64_t i = 0; i < ndim; ++i) shape[i] = hp[2 + i];

        auto staged = torch::empty(shape, torch::TensorOptions().dtype(dt).device(detail::active_device()));
        detail::nccl_check(ncclGroupStart(), "ncclGroupStart(scatter-recv-data)");
        detail::nccl_check(
            ncclRecv(staged.data_ptr(), static_cast<size_t>(staged.numel()),
                     detail::to_nccl_dtype(dt), src, s.comm, stream),
            "ncclRecv(scatter-data)");
        detail::nccl_check(ncclGroupEnd(), "ncclGroupEnd(scatter-recv-data)");
        detail::stream_sync();

        detail::copy_back(staged, output);
    }
}

} // namespace distdl::distributed::nccl_backend

#else

namespace distdl::distributed::nccl_backend
{
enum class ReduceOp { Sum, Product, Min, Max };
inline void init() { throw std::runtime_error("NCCL backend not compiled in"); }
inline void finalize() {}
inline bool is_initialized() { return false; }
inline int get_rank() { return 0; }
inline int get_local_rank() { return 0; }
inline int get_world_size() { return 1; }
inline void send(const torch::Tensor&, int, int = 0) { throw std::runtime_error("NCCL backend not compiled in"); }
inline void recv(torch::Tensor&, int, int = 0) { throw std::runtime_error("NCCL backend not compiled in"); }
inline void broadcast(torch::Tensor&, int) { throw std::runtime_error("NCCL backend not compiled in"); }
inline void reduce(const torch::Tensor&, torch::Tensor&, int, ReduceOp = ReduceOp::Sum) { throw std::runtime_error("NCCL backend not compiled in"); }
inline void gather(const torch::Tensor&, std::vector<torch::Tensor>&, int) { throw std::runtime_error("NCCL backend not compiled in"); }
inline void scatter(torch::Tensor&, const std::vector<torch::Tensor>&, int) { throw std::runtime_error("NCCL backend not compiled in"); }
inline void allreduce(torch::Tensor&, ReduceOp = ReduceOp::Sum) { throw std::runtime_error("NCCL backend not compiled in"); }
inline void barrier() { throw std::runtime_error("NCCL backend not compiled in"); }
}

#endif