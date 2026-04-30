#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "mpi_ops.h"
#include "nccl_ops.h"

namespace distdl::distributed
{
enum class Backend { MPI, NCCL };

enum class ReduceOp { Sum, Product, Min, Max };

namespace detail
{
inline Backend& active_backend()
{
    static Backend b = Backend::MPI;
    return b;
}

inline bool& backend_chosen()
{
    static bool c = false;
    return c;
}

inline mpi_backend::ReduceOp to_mpi_op(ReduceOp op)
{
    switch (op)
    {
    case ReduceOp::Sum:     return mpi_backend::ReduceOp::Sum;
    case ReduceOp::Product: return mpi_backend::ReduceOp::Product;
    case ReduceOp::Min:     return mpi_backend::ReduceOp::Min;
    case ReduceOp::Max:     return mpi_backend::ReduceOp::Max;
    }
    throw std::runtime_error("bad reduce op");
}

inline nccl_backend::ReduceOp to_nccl_op(ReduceOp op)
{
    switch (op)
    {
    case ReduceOp::Sum:     return nccl_backend::ReduceOp::Sum;
    case ReduceOp::Product: return nccl_backend::ReduceOp::Product;
    case ReduceOp::Min:     return nccl_backend::ReduceOp::Min;
    case ReduceOp::Max:     return nccl_backend::ReduceOp::Max;
    }
    throw std::runtime_error("bad reduce op");
}

inline void ensure_chosen()
{
    if (!backend_chosen())
        throw std::runtime_error("distdl::distributed: init(Backend) was not called");
}
}

inline Backend parse_backend(const std::string& s)
{
    if (s == "mpi" || s == "MPI")   return Backend::MPI;
    if (s == "nccl" || s == "NCCL") return Backend::NCCL;
    throw std::runtime_error("Unknown backend '" + s + "' (expected 'mpi' or 'nccl')");
}

inline const char* backend_name(Backend b)
{
    return b == Backend::NCCL ? "nccl" : "mpi";
}

inline void init(Backend b)
{
    detail::active_backend() = b;
    detail::backend_chosen() = true;
    if (b == Backend::MPI) mpi_backend::init();
    else                   nccl_backend::init();
}

inline void finalize()
{
    if (!detail::backend_chosen()) return;
    if (detail::active_backend() == Backend::MPI) mpi_backend::finalize();
    else                                          nccl_backend::finalize();
    detail::backend_chosen() = false;
}

inline Backend current_backend()
{
    detail::ensure_chosen();
    return detail::active_backend();
}

inline bool is_initialized()
{
    if (!detail::backend_chosen()) return false;
    return detail::active_backend() == Backend::MPI
        ? mpi_backend::is_initialized()
        : nccl_backend::is_initialized();
}

inline int get_rank()
{
    detail::ensure_chosen();
    return detail::active_backend() == Backend::MPI
        ? mpi_backend::get_rank() : nccl_backend::get_rank();
}

inline int get_world_size()
{
    detail::ensure_chosen();
    return detail::active_backend() == Backend::MPI
        ? mpi_backend::get_world_size() : nccl_backend::get_world_size();
}

inline int get_local_rank()
{
    detail::ensure_chosen();
    return detail::active_backend() == Backend::MPI
        ? mpi_backend::get_local_rank() : nccl_backend::get_local_rank();
}

inline void send(const torch::Tensor& t, int dst, int tag = 0)
{
    detail::ensure_chosen();
    if (detail::active_backend() == Backend::MPI) mpi_backend::send(t, dst, tag);
    else                                          nccl_backend::send(t, dst, tag);
}

inline void recv(torch::Tensor& t, int src, int tag = 0)
{
    detail::ensure_chosen();
    if (detail::active_backend() == Backend::MPI) mpi_backend::recv(t, src, tag);
    else                                          nccl_backend::recv(t, src, tag);
}

inline void broadcast(torch::Tensor& t, int src)
{
    detail::ensure_chosen();
    if (detail::active_backend() == Backend::MPI) mpi_backend::broadcast(t, src);
    else                                          nccl_backend::broadcast(t, src);
}

inline void reduce(const torch::Tensor& in, torch::Tensor& out, int dst, ReduceOp op = ReduceOp::Sum)
{
    detail::ensure_chosen();
    if (detail::active_backend() == Backend::MPI)
        mpi_backend::reduce(in, out, dst, detail::to_mpi_op(op));
    else
        nccl_backend::reduce(in, out, dst, detail::to_nccl_op(op));
}

inline void gather(const torch::Tensor& in, std::vector<torch::Tensor>& list, int dst)
{
    detail::ensure_chosen();
    if (detail::active_backend() == Backend::MPI) mpi_backend::gather(in, list, dst);
    else                                          nccl_backend::gather(in, list, dst);
}

inline void scatter(torch::Tensor& out, const std::vector<torch::Tensor>& list, int src)
{
    detail::ensure_chosen();
    if (detail::active_backend() == Backend::MPI) mpi_backend::scatter(out, list, src);
    else                                          nccl_backend::scatter(out, list, src);
}

inline void allreduce(torch::Tensor& t, ReduceOp op = ReduceOp::Sum)
{
    detail::ensure_chosen();
    if (detail::active_backend() == Backend::MPI)
        mpi_backend::allreduce(t, detail::to_mpi_op(op));
    else
        nccl_backend::allreduce(t, detail::to_nccl_op(op));
}

inline void barrier()
{
    detail::ensure_chosen();
    if (detail::active_backend() == Backend::MPI) mpi_backend::barrier();
    else                                          nccl_backend::barrier();
}

} // namespace distdl::distributed