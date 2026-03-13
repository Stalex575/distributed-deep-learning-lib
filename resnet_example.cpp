#include <torch/torch.h>
#include <iostream>
#include <string>

#include "cifar10_dataset.h"

struct ResidualBlockImpl : torch::nn::Module
{
    torch::nn::Conv2d conv1{ nullptr };
    torch::nn::BatchNorm2d bn1{ nullptr };
    torch::nn::Conv2d conv2{ nullptr };
    torch::nn::BatchNorm2d bn2{ nullptr };
    torch::nn::Sequential shortcut{ nullptr };

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

    ResNetImpl(int64_t num_classes = 10)
    {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).stride(1).padding(1).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));

        layer1 = register_module("layer1", _make_layer(64, 2, 1));
        layer2 = register_module("layer2", _make_layer(128, 2, 2));
        layer3 = register_module("layer3", _make_layer(256, 2, 2));
        layer4 = register_module("layer4", _make_layer(512, 2, 2));

        avgpool = register_module("avgpool", torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
        fc = register_module("fc", torch::nn::Linear(512, num_classes));
    }

    torch::nn::Sequential _make_layer(int64_t out_channels, int64_t num_blocks, int64_t stride)
    {
        torch::nn::Sequential layers;
        layers->push_back(ResidualBlock(in_channels, out_channels, stride));
        in_channels = out_channels;
        for (int i = 1; i < num_blocks; ++i)
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

int main()
{
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

    const std::string data_path = "./data/cifar-10-batches-bin"; 

    auto train_dataset = CIFAR10(data_path, CIFAR10::Mode::kTrain)
        .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
        .map(torch::data::transforms::Stack<>()
    );
    
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), torch::data::DataLoaderOptions().batch_size(128)
    );

    auto test_dataset = CIFAR10(data_path, CIFAR10::Mode::kTest)
        .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
        .map(torch::data::transforms::Stack<>());
    
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), torch::data::DataLoaderOptions().batch_size(128)
    );

    ResNet model(10);
    model->to(device);

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.001));

    int epochs = 5;
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        model->train();
        double sum_loss = 0.0;
        int batch_idx = 0;

        for (auto& batch : *train_loader)
        {
            auto inputs = batch.data.to(device);
            auto labels = batch.target.to(device);

            optimizer.zero_grad();
            auto outputs = model->forward(inputs);
            auto loss = torch::nn::functional::cross_entropy(outputs, labels);
            loss.backward();
            optimizer.step();

            sum_loss += loss.item<double>();
            if ((batch_idx + 1) % 10 == 0)
            {
                std::cout << "[Epoch " << (epoch + 1) << ", Batch " << (batch_idx + 1) << "] Loss: " << (sum_loss / 10.0) << std::endl;
                sum_loss = 0.0;
            }
            ++batch_idx;
        }
    }

    std::cout << "\nGradient matrices:\n";
    for (const auto& pair : model->named_parameters())
    {
        if (pair.value().requires_grad() && pair.value().grad().defined())
        {
            std::cout << pair.key() << ": " << pair.value().grad().sizes() << "\n";
        }
    }

    model->eval();
    int correct = 0;
    int total = 0;

    {
        torch::NoGradGuard no_grad;
        for (const auto& batch : *test_loader) {
            auto inputs = batch.data.to(device);
            auto labels = batch.target.to(device);

            auto outputs = model->forward(inputs);
            auto prediction = outputs.argmax(1);
            
            total += labels.sizes()[0];
            correct += prediction.eq(labels).sum().item<int>();
        }
    }

    std::cout << "\nAccuracy on test set: " << (double)correct / total * 100.0 << "%" << std::endl;

    return 0;
}
