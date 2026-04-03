#include <torch/torch.h>
#include <iostream>
#include <string>

#include "dist_deep_learning_lib.h"

int main()
{
    distdl::init_distributed_training();
    const int rank = distdl::distributed::get_rank();
    const int world = distdl::distributed::get_world_size();

    try
    {
        torch::manual_seed(1337 + rank);

        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available())
        {
            const int gpu_count = torch::cuda::device_count();
            const int gpu_index = gpu_count > 0 ? (rank % gpu_count) : 0;
            device = torch::Device(torch::kCUDA, gpu_index);
        }

        if (rank == 0)
        {
            std::cout << "World size: " << world << std::endl;
        }
        std::cout << "[Rank " << rank << "] Using device: "
                  << (device.is_cuda() ? "CUDA:" + std::to_string(device.index()) : "CPU")
                  << std::endl;

        const std::string data_path = "./data/cifar-10-batches-bin";

        auto train_dataset = CIFAR10(data_path, CIFAR10::Mode::kTrain)
            .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
            .map(torch::data::transforms::Stack<>());

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
        distdl::broadcast_model_state(*model, 0);

        torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.001));

        const int epochs = 5;
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

                distdl::average_gradients(*model, 0);
                optimizer.step();

                const auto mean_loss = distdl::average_scalar_loss(loss.detach(), 0);
                sum_loss += mean_loss.item<double>();
                if (rank == 0 && (batch_idx + 1) % 10 == 0)
                {
                    std::cout << "[Epoch " << (epoch + 1)
                              << ", Batch " << (batch_idx + 1)
                              << "] Loss: " << (sum_loss / 10.0)
                              << std::endl;
                    sum_loss = 0.0;
                }
                ++batch_idx;
            }
        }

        if (rank == 0)
        {
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
                for (const auto& batch : *test_loader)
                {
                    auto inputs = batch.data.to(device);
                    auto labels = batch.target.to(device);

                    auto outputs = model->forward(inputs);
                    auto prediction = outputs.argmax(1);

                    total += labels.sizes()[0];
                    correct += prediction.eq(labels).sum().item<int>();
                }
            }

            std::cout << "\nAccuracy on test set: " << (double)correct / total * 100.0 << "%" << std::endl;
        }

        distdl::finalize_distributed_training();
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[Rank " << rank << "] Error: " << ex.what() << std::endl;
        try
        {
            distdl::finalize_distributed_training();
        }
        catch (...)
        {
        }
        return 1;
    }
}
