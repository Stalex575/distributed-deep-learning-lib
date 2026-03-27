#pragma once

#include <cstdint>

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
} // namespace distdl
