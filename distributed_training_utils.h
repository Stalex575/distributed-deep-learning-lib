#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "distributed_ops.h"

namespace distdl
{
inline bool distributed_active()
{
    return distributed::is_initialized() && distributed::get_world_size() > 1;
}

inline void init_distributed_training()
{
    distributed::init();
}

inline void finalize_distributed_training()
{
    if (!distributed::is_initialized())
    {
        return;
    }

    if (distributed_active())
    {
        distributed::barrier();
    }
    distributed::finalize();
}

inline void broadcast_model_state(torch::nn::Module& module, int src_rank = 0)
{
    if (!distributed_active())
    {
        return;
    }

    torch::NoGradGuard no_grad;

    for (auto& parameter : module.parameters())
    {
        if (!parameter.defined())
        {
            continue;
        }

        torch::Tensor synced = parameter.detach().clone();
        distributed::broadcast(synced, src_rank);
        parameter.copy_(synced.to(parameter.device(), parameter.scalar_type()));
    }

    for (auto& buffer : module.buffers())
    {
        if (!buffer.defined())
        {
            continue;
        }

        torch::Tensor synced = buffer.detach().clone();
        distributed::broadcast(synced, src_rank);
        buffer.copy_(synced.to(buffer.device(), buffer.scalar_type()));
    }

    distributed::barrier();
}

inline void average_gradients(torch::nn::Module& module)
{
    if (!distributed_active())
    {
        return;
    }

    const double world = static_cast<double>(distributed::get_world_size());
    torch::NoGradGuard no_grad;

    for (auto& parameter : module.parameters())
    {
        torch::Tensor grad = parameter.grad();
        if (!grad.defined())
        {
            continue;
        }

        torch::Tensor summed = grad.detach().clone();
        distributed::allreduce(summed, distributed::ReduceOp::Sum);
        summed.div_(world);
        grad.copy_(summed.to(grad.device(), grad.scalar_type()));
    }
}

inline torch::Tensor average_scalar_loss(const torch::Tensor& local_loss)
{
    if (local_loss.numel() != 1)
    {
        throw std::runtime_error("average_scalar_loss expects a scalar tensor");
    }

    if (!distributed_active())
    {
        return local_loss.detach().clone();
    }

    const double world = static_cast<double>(distributed::get_world_size());

    torch::Tensor summed = local_loss.detach().clone();
    distributed::allreduce(summed, distributed::ReduceOp::Sum);
    summed.div_(world);
    return summed;
}

inline torch::Tensor sum_scalar_tensor(const torch::Tensor& local_value)
{
    if (local_value.numel() != 1)
    {
        throw std::runtime_error("sum_scalar_tensor expects a scalar tensor");
    }

    if (!distributed_active())
    {
        return local_value.detach().clone();
    }

    torch::Tensor summed = local_value.detach().clone();
    distributed::allreduce(summed, distributed::ReduceOp::Sum);
    return summed;
}

inline std::pair<torch::Tensor, torch::Tensor> shard_batch_for_rank(
    const torch::Tensor& batch_data,
    const torch::Tensor& batch_target,
    int rank,
    int world
)
{
    if (batch_data.size(0) != batch_target.size(0))
    {
        throw std::runtime_error("shard_batch_for_rank expects matching batch dimension");
    }

    if (world <= 1)
    {
        return {batch_data, batch_target};
    }

    if (rank < 0 || rank >= world)
    {
        throw std::runtime_error("shard_batch_for_rank got invalid rank/world values");
    }

    const int64_t batch_size = batch_data.size(0);
    const int64_t start = (batch_size * static_cast<int64_t>(rank)) / static_cast<int64_t>(world);
    const int64_t end = (batch_size * static_cast<int64_t>(rank + 1)) / static_cast<int64_t>(world);

    if (start == end)
    {
        const auto data_shape = batch_data.sizes().vec();
        const auto target_shape = batch_target.sizes().vec();

        std::vector<int64_t> local_data_shape = data_shape;
        std::vector<int64_t> local_target_shape = target_shape;
        local_data_shape[0] = 0;
        local_target_shape[0] = 0;

        torch::Tensor empty_data = torch::empty(local_data_shape, batch_data.options());
        torch::Tensor empty_target = torch::empty(local_target_shape, batch_target.options());
        return {empty_data, empty_target};
    }

    torch::Tensor local_data = batch_data.slice(0, start, end);
    torch::Tensor local_target = batch_target.slice(0, start, end);
    return {local_data, local_target};
}

inline bool rank_owns_batch(std::size_t global_batch_index)
{
    if (!distributed_active())
    {
        return true;
    }

    const std::size_t world = static_cast<std::size_t>(distributed::get_world_size());
    const std::size_t rank = static_cast<std::size_t>(distributed::get_rank());
    return (global_batch_index % world) == rank;
}
} // namespace distdl
