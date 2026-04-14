#pragma once

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "serializer.h"

#if defined(DISTDL_HAS_MPI) && DISTDL_HAS_MPI
#include <mpi.h>
#endif

namespace distdl::distributed
{
enum class ReduceOp
{
    Sum,
    Product,
    Min,
    Max,
};

#if !defined(DISTDL_HAS_MPI) || !(DISTDL_HAS_MPI)

inline void init() {}
inline void finalize() {}
inline bool is_initialized() { return false; }
inline int get_rank() { return 0; }
inline int get_local_rank() { return 0; }
inline int get_world_size() { return 1; }

inline void send(const torch::Tensor&, int, int = 0)
{
    throw std::runtime_error("MPI support is not enabled (rebuild with MPI available)");
}

inline void recv(torch::Tensor&, int, int = 0)
{
    throw std::runtime_error("MPI support is not enabled (rebuild with MPI available)");
}

inline void broadcast(torch::Tensor&, int)
{
    throw std::runtime_error("MPI support is not enabled (rebuild with MPI available)");
}

inline void reduce(const torch::Tensor&, torch::Tensor&, int, ReduceOp = ReduceOp::Sum)
{
    throw std::runtime_error("MPI support is not enabled (rebuild with MPI available)");
}

inline void gather(const torch::Tensor&, std::vector<torch::Tensor>&, int)
{
    throw std::runtime_error("MPI support is not enabled (rebuild with MPI available)");
}

inline void scatter(torch::Tensor&, const std::vector<torch::Tensor>&, int)
{
    throw std::runtime_error("MPI support is not enabled (rebuild with MPI available)");
}

inline void allreduce(torch::Tensor&, ReduceOp = ReduceOp::Sum)
{
    throw std::runtime_error("MPI support is not enabled (rebuild with MPI available)");
}

inline void barrier()
{
    throw std::runtime_error("MPI support is not enabled (rebuild with MPI available)");
}

#else

namespace detail
{
inline void mpi_check(int code, const char* where)
{
    if (code != MPI_SUCCESS)
    {
        throw std::runtime_error(std::string("MPI error in ") + where + ", code=" + std::to_string(code));
    }
}

inline int to_int_count(int64_t n, const char* what)
{
    if (n > static_cast<int64_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error(std::string(what) + " is too large for MPI int count");
    }
    return static_cast<int>(n);
}

inline int world_rank()
{
    int rank = 0;
    mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    return rank;
}

inline int world_size()
{
    int world = 1;
    mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &world), "MPI_Comm_size");
    return world;
}

inline int try_parse_env_int(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return -1;
    }

    try
    {
        return std::stoi(value);
    }
    catch (...)
    {
        return -1;
    }
}

inline int local_rank_from_env_or_global()
{
    const int local_rank = [&]() {
        const int ompi = try_parse_env_int("OMPI_COMM_WORLD_LOCAL_RANK");
        if (ompi >= 0)
        {
            return ompi;
        }

        const int mvapich = try_parse_env_int("MV2_COMM_WORLD_LOCAL_RANK");
        if (mvapich >= 0)
        {
            return mvapich;
        }

        const int intel_mpi = try_parse_env_int("MPI_LOCALRANKID");
        if (intel_mpi >= 0)
        {
            return intel_mpi;
        }

        const int slurm = try_parse_env_int("SLURM_LOCALID");
        if (slurm >= 0)
        {
            return slurm;
        }

        return -1;
    }();

    if (local_rank >= 0)
    {
        return local_rank;
    }

    return world_rank();
}

inline void send_buffer(const std::vector<std::byte>& buffer, int dst, int tag)
{
    const std::uint64_t n = static_cast<std::uint64_t>(buffer.size());
    mpi_check(MPI_Send(&n, 1, MPI_UNSIGNED_LONG_LONG, dst, tag, MPI_COMM_WORLD), "MPI_Send(size)");
    if (!buffer.empty())
    {
        const int count = to_int_count(static_cast<int64_t>(buffer.size()), "send byte count");
        mpi_check(MPI_Send(buffer.data(), count, MPI_BYTE, dst, tag + 1, MPI_COMM_WORLD), "MPI_Send(data)");
    }
}

inline std::vector<std::byte> recv_buffer(int src, int tag)
{
    std::uint64_t n = 0;
    mpi_check(MPI_Recv(&n, 1, MPI_UNSIGNED_LONG_LONG, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE), "MPI_Recv(size)");
    std::vector<std::byte> buffer(static_cast<size_t>(n));
    if (n > 0)
    {
        const int count = to_int_count(static_cast<int64_t>(n), "recv byte count");
        mpi_check(MPI_Recv(buffer.data(), count, MPI_BYTE, src, tag + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE), "MPI_Recv(data)");
    }
    return buffer;
}

inline void broadcast_buffer(std::vector<std::byte>& buffer, int src)
{
    std::uint64_t n = static_cast<std::uint64_t>(buffer.size());
    mpi_check(MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG_LONG, src, MPI_COMM_WORLD), "MPI_Bcast(size)");
    if (world_rank() != src)
    {
        buffer.resize(static_cast<size_t>(n));
    }

    if (n > 0)
    {
        const int count = to_int_count(static_cast<int64_t>(n), "broadcast byte count");
        mpi_check(MPI_Bcast(buffer.data(), count, MPI_BYTE, src, MPI_COMM_WORLD), "MPI_Bcast(data)");
    }
}

inline MPI_Datatype dtype_to_mpi(torch::ScalarType dt)
{
    switch (dt)
    {
    case torch::kFloat32:
        return MPI_FLOAT;
    case torch::kFloat64:
        return MPI_DOUBLE;
    case torch::kInt32:
        return MPI_INT;
    case torch::kInt64:
        return MPI_LONG_LONG;
    case torch::kUInt8:
        return MPI_UNSIGNED_CHAR;
    case torch::kInt8:
        return MPI_SIGNED_CHAR;
    case torch::kBool:
        return MPI_C_BOOL;
    default:
        throw std::runtime_error("Unsupported dtype for reduce");
    }
}

inline MPI_Op reduce_to_mpi(ReduceOp op)
{
    switch (op)
    {
    case ReduceOp::Sum:
        return MPI_SUM;
    case ReduceOp::Product:
        return MPI_PROD;
    case ReduceOp::Min:
        return MPI_MIN;
    case ReduceOp::Max:
        return MPI_MAX;
    }
    throw std::runtime_error("Unsupported reduce op");
}

inline bool initialized_by_distdl = false;
} // namespace detail

inline void init()
{
    int initialized = 0;
    detail::mpi_check(MPI_Initialized(&initialized), "MPI_Initialized");
    if (!initialized)
    {
        int argc = 0;
        char** argv = nullptr;
        int provided = 0;
        detail::mpi_check(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided), "MPI_Init_thread");
        detail::initialized_by_distdl = true;
    }
}

inline void finalize()
{
    int initialized = 0;
    int finalized = 0;
    detail::mpi_check(MPI_Initialized(&initialized), "MPI_Initialized");
    detail::mpi_check(MPI_Finalized(&finalized), "MPI_Finalized");
    if (initialized && !finalized && detail::initialized_by_distdl)
    {
        detail::mpi_check(MPI_Finalize(), "MPI_Finalize");
        detail::initialized_by_distdl = false;
    }
}

inline bool is_initialized()
{
    int initialized = 0;
    detail::mpi_check(MPI_Initialized(&initialized), "MPI_Initialized");
    return initialized != 0;
}

inline int get_rank()
{
    if (!is_initialized())
    {
        return 0;
    }
    return detail::world_rank();
}

inline int get_world_size()
{
    if (!is_initialized())
    {
        return 1;
    }
    return detail::world_size();
}

inline int get_local_rank()
{
    if (!is_initialized())
    {
        return 0;
    }
    return detail::local_rank_from_env_or_global();
}

inline void send(const torch::Tensor& tensor, int dst, int tag = 0)
{
    detail::send_buffer(distdl::serialize(tensor), dst, tag);
}

inline void recv(torch::Tensor& tensor, int src, int tag = 0)
{
    torch::Tensor out = distdl::deserialize(detail::recv_buffer(src, tag));
    if (tensor.defined())
    {
        out = out.to(tensor.device(), tensor.scalar_type());
    }
    tensor = out;
}

inline void broadcast(torch::Tensor& tensor, int src)
{
    std::vector<std::byte> buffer;
    if (detail::world_rank() == src)
    {
        buffer = distdl::serialize(tensor);
    }
    detail::broadcast_buffer(buffer, src);

    torch::Tensor out = distdl::deserialize(buffer);
    if (tensor.defined())
    {
        out = out.to(tensor.device(), tensor.scalar_type());
    }
    tensor = out;
}

inline void reduce(const torch::Tensor& input, torch::Tensor& output, int dst, ReduceOp op = ReduceOp::Sum)
{
    if (input.sizes() != output.sizes() || input.scalar_type() != output.scalar_type())
    {
        throw std::runtime_error("reduce: input/output shape and dtype must match");
    }

    const torch::Tensor cpu_in = input.to(torch::kCPU).contiguous();
    torch::Tensor cpu_out = output.to(torch::kCPU).contiguous();
    const int count = detail::to_int_count(cpu_in.numel(), "reduce element count");

    detail::mpi_check(
        MPI_Reduce(
            cpu_in.data_ptr(),
            cpu_out.data_ptr(),
            count,
            detail::dtype_to_mpi(cpu_in.scalar_type()),
            detail::reduce_to_mpi(op),
            dst,
            MPI_COMM_WORLD
        ),
        "MPI_Reduce"
    );
    if (detail::world_rank() == dst)
    {
        output = cpu_out.to(output.device(), output.scalar_type());
    }
}

inline void gather(const torch::Tensor& input, std::vector<torch::Tensor>& gather_list, int dst)
{
    const int rank = detail::world_rank();
    const int world = detail::world_size();
    const int base_tag = 20000;

    if (rank == dst)
    {
        if (static_cast<int>(gather_list.size()) != world)
        {
            throw std::runtime_error("gather: gather_list size must match world size on destination rank");
        }

        for (int r = 0; r < world; ++r)
        {
            torch::Tensor out = (r == dst)
                ? distdl::deserialize(distdl::serialize(input))
                : distdl::deserialize(detail::recv_buffer(r, base_tag));

            if (gather_list[static_cast<size_t>(r)].defined())
            {
                out = out.to(
                    gather_list[static_cast<size_t>(r)].device(),
                    gather_list[static_cast<size_t>(r)].scalar_type()
                );
            }
            gather_list[static_cast<size_t>(r)] = out;
        }
    }
    else
    {
        detail::send_buffer(distdl::serialize(input), dst, base_tag);
    }
}

inline void scatter(torch::Tensor& output, const std::vector<torch::Tensor>& scatter_list, int src)
{
    const int rank = detail::world_rank();
    const int world = detail::world_size();
    const int base_tag = 21000;

    if (rank == src)
    {
        if (static_cast<int>(scatter_list.size()) != world)
        {
            throw std::runtime_error("scatter: scatter_list size must match world size on source rank");
        }

        for (int r = 0; r < world; ++r)
        {
            if (r == src)
            {
                output = scatter_list[static_cast<size_t>(r)];
                continue;
            }
            detail::send_buffer(distdl::serialize(scatter_list[static_cast<size_t>(r)]), r, base_tag);
        }
    }
    else
    {
        torch::Tensor out = distdl::deserialize(detail::recv_buffer(src, base_tag));
        if (output.defined())
        {
            out = out.to(output.device(), output.scalar_type());
        }
        output = out;
    }
}

inline void allreduce(torch::Tensor& tensor, ReduceOp op = ReduceOp::Sum)
{
    const torch::Tensor cpu = tensor.to(torch::kCPU).contiguous();
    const int count = detail::to_int_count(cpu.numel(), "allreduce element count");
    const MPI_Datatype mpi_dt = detail::dtype_to_mpi(cpu.scalar_type());
    const MPI_Op mpi_op = detail::reduce_to_mpi(op);

#if defined(DISTDL_USE_MPI_ALLREDUCE) && DISTDL_USE_MPI_ALLREDUCE
    torch::Tensor result = torch::empty_like(cpu);
    detail::mpi_check(
        MPI_Allreduce(
            cpu.data_ptr(),
            result.data_ptr(),
            count,
            mpi_dt,
            mpi_op,
            MPI_COMM_WORLD
        ),
        "MPI_Allreduce"
    );
#else
    // Fallback: reduce to rank 0, then broadcast.
    constexpr int root = 0;
    torch::Tensor result = torch::zeros_like(cpu);
    detail::mpi_check(
        MPI_Reduce(
            cpu.data_ptr(),
            result.data_ptr(),
            count,
            mpi_dt,
            mpi_op,
            root,
            MPI_COMM_WORLD
        ),
        "MPI_Reduce(allreduce fallback)"
    );
    detail::mpi_check(
        MPI_Bcast(
            result.data_ptr(),
            count,
            mpi_dt,
            root,
            MPI_COMM_WORLD
        ),
        "MPI_Bcast(allreduce fallback)"
    );
#endif

    tensor = result.to(tensor.device(), tensor.scalar_type());
}

inline void barrier()
{
    detail::mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
}

#endif

} // namespace distdl::distributed
