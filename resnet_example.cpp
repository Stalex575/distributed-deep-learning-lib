#include <torch/torch.h>
#include <iostream>
#include <string>

#include "dist_deep_learning_lib.h"

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

    distdl::ResNet model(10);
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
