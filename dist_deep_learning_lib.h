#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <torch/torch.h>

#include "cifar10_dataset.h"
#include "distributed_ops.h"

namespace distdl
{
struct ResidualBlockImpl : torch::nn::Module
{
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::BatchNorm2d bn2{nullptr};
    torch::nn::Sequential shortcut{nullptr};

    ResidualBlockImpl(int64_t in_channels, int64_t out_channels, int64_t stride = 1)
    {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3)
            .stride(stride)
            .padding(1)
            .bias(false))
        );
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channels));

        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3)
            .stride(1)
            .padding(1)
            .bias(false))
        );
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));

        if (stride != 1 || in_channels != out_channels)
        {
            shortcut = register_module("shortcut", torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1)
                    .stride(stride)
                    .bias(false)
                ),
                torch::nn::BatchNorm2d(out_channels)
            ));
        }
        else
        {
            shortcut = register_module("shortcut", torch::nn::Sequential());
        }
    }

    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor identity = shortcut->is_empty() ? x : shortcut->forward(x);
        torch::Tensor out = conv1->forward(x);

        out = bn1->forward(out);
        out = torch::relu(out);
        out = conv2->forward(out);
        out = bn2->forward(out);
        out += identity;
        out = torch::relu(out);

        return out;
    }
};
TORCH_MODULE(ResidualBlock);

struct ResNetImpl : torch::nn::Module
{
    int64_t in_channels = 64;
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::Sequential layer1{nullptr};
    torch::nn::Sequential layer2{nullptr};
    torch::nn::Sequential layer3{nullptr};
    torch::nn::Sequential layer4{nullptr};
    torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
    torch::nn::Linear fc{nullptr};

    explicit ResNetImpl(int64_t num_classes = 10)
    {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).stride(1).padding(1).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));

        layer1 = register_module("layer1", make_layer(64, 2, 1));
        layer2 = register_module("layer2", make_layer(128, 2, 2));
        layer3 = register_module("layer3", make_layer(256, 2, 2));
        layer4 = register_module("layer4", make_layer(512, 2, 2));

        avgpool = register_module("avgpool", torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
        fc = register_module("fc", torch::nn::Linear(512, num_classes));
    }

    torch::nn::Sequential make_layer(int64_t out_channels, int64_t num_blocks, int64_t stride)
    {
        torch::nn::Sequential layers;
        layers->push_back(ResidualBlock(in_channels, out_channels, stride));
        in_channels = out_channels;

        for (int64_t i = 1; i < num_blocks; ++i)
        {
            layers->push_back(ResidualBlock(in_channels, out_channels, 1));
        }

        return layers;
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = conv1->forward(x);
        x = bn1->forward(x);
        x = torch::relu(x);
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);
        x = avgpool->forward(x);
        x = torch::flatten(x, 1);
        x = fc->forward(x);
        return x;
    }
};
TORCH_MODULE(ResNet);

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

inline void average_gradients(torch::nn::Module& module, int dst_rank = 0)
{
    if (!distributed_active())
    {
        return;
    }

    const int world = distributed::get_world_size();
    const int rank = distributed::get_rank();
    torch::NoGradGuard no_grad;

    for (auto& parameter : module.parameters())
    {
        torch::Tensor grad = parameter.grad();
        if (!grad.defined())
        {
            continue;
        }

        torch::Tensor reduced = torch::zeros_like(grad);
        distributed::reduce(grad, reduced, dst_rank, distributed::ReduceOp::Sum);

        if (rank == dst_rank)
        {
            reduced.div_(static_cast<double>(world));
        }

        distributed::broadcast(reduced, dst_rank);
        grad.copy_(reduced.to(grad.device(), grad.scalar_type()));
    }
}

inline torch::Tensor average_scalar_loss(const torch::Tensor& local_loss, int dst_rank = 0)
{
    if (local_loss.numel() != 1)
    {
        throw std::runtime_error("average_scalar_loss expects a scalar tensor");
    }

    if (!distributed_active())
    {
        return local_loss.detach().clone();
    }

    const int world = distributed::get_world_size();
    const int rank = distributed::get_rank();

    torch::Tensor reduced = torch::zeros_like(local_loss);
    distributed::reduce(local_loss.detach(), reduced, dst_rank, distributed::ReduceOp::Sum);

    if (rank == dst_rank)
    {
        reduced.div_(static_cast<double>(world));
    }

    distributed::broadcast(reduced, dst_rank);
    return reduced;
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
