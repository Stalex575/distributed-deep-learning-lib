#include <torch/torch.h>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <string>

#include "dist_deep_learning_lib.h"

namespace
{
distdl::distributed::Backend parse_args(int argc, char** argv)
{
    std::string choice = "mpi";
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc)
        {
            choice = argv[i + 1];
            ++i;
        }
        else if (std::strncmp(argv[i], "--backend=", 10) == 0)
        {
            choice = argv[i] + 10;
        }
    }
    return distdl::distributed::parse_backend(choice);
}
}

int main(int argc, char** argv)
{
    distdl::distributed::Backend backend = distdl::distributed::Backend::MPI;
    try { backend = parse_args(argc, argv); }
    catch (const std::exception& ex)
    {
        std::cerr << "Argument error: " << ex.what() << std::endl;
        return 1;
    }

    distdl::init_distributed_training(backend);
    const int rank = distdl::distributed::get_rank();
    const int local_rank = distdl::distributed::get_local_rank();
    const int world = distdl::distributed::get_world_size();

    try
    {
        torch::manual_seed(1337);

        torch::Device device(torch::kCPU);
        if (backend == distdl::distributed::Backend::NCCL)
        {
            const int gpu_count = torch::cuda::device_count();
            if (gpu_count <= 0)
                throw std::runtime_error("NCCL backend requested but no CUDA devices visible");
            device = torch::Device(torch::kCUDA, local_rank % gpu_count);
        }
        else if (torch::cuda::is_available())
        {
            const int gpu_count = torch::cuda::device_count();
            const int gpu_index = gpu_count > 0 ? (local_rank % gpu_count) : 0;
            device = torch::Device(torch::kCUDA, gpu_index);
        }

        if (rank == 0)
        {
            std::cout << "Backend: " << distdl::distributed::backend_name(backend) << std::endl;
            std::cout << "World size: " << world << std::endl;
        }
        std::cout << "[Rank " << rank << "] device: "
                  << (device.is_cuda() ? "CUDA:" + std::to_string(device.index()) : "CPU")
                  << std::endl;

        const std::string data_path = "./data/cifar-10-batches-bin";

        auto train_dataset = CIFAR10(data_path, CIFAR10::Mode::kTrain)
            .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
            .map(torch::data::transforms::Stack<>());

        auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset),
            torch::data::DataLoaderOptions().batch_size(128).drop_last(true));

        auto test_dataset = CIFAR10(data_path, CIFAR10::Mode::kTest)
            .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
            .map(torch::data::transforms::Stack<>());

        auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(test_dataset), torch::data::DataLoaderOptions().batch_size(128));

        distdl::ResNet model(10);
        model->to(device);
        distdl::broadcast_model_state(*model, 0);

        torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.001));

        const int epochs = 1;
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            model->train();
            double sum_loss = 0.0;
            int batch_idx = 0;
            torch::manual_seed(1337 + epoch);

            for (auto& batch : *train_loader)
            {
                auto local_batch = distdl::shard_batch_for_rank(batch.data, batch.target, rank, world);
                auto inputs = local_batch.first.to(device);
                auto labels = local_batch.second.to(device);
                if (inputs.size(0) == 0) continue;

                optimizer.zero_grad();
                auto outputs = model->forward(inputs);
                auto loss = torch::nn::functional::cross_entropy(outputs, labels);
                loss.backward();

                distdl::average_gradients(*model);
                optimizer.step();

                const auto mean_loss = distdl::average_scalar_loss(loss.detach());
                sum_loss += mean_loss.item<double>();
                if (rank == 0 && (batch_idx + 1) % 10 == 0)
                {
                    std::cout << "[Epoch " << (epoch + 1)
                              << ", Batch " << (batch_idx + 1)
                              << "] Loss: " << (sum_loss / 10.0) << std::endl;
                    sum_loss = 0.0;
                }
                ++batch_idx;
            }
        }

        if (distdl::distributed_active()) distdl::distributed::barrier();

        model->eval();
        std::int64_t local_correct = 0;
        std::int64_t local_total = 0;
        {
            torch::NoGradGuard no_grad;
            for (const auto& batch : *test_loader)
            {
                auto local_batch = distdl::shard_batch_for_rank(batch.data, batch.target, rank, world);
                auto inputs = local_batch.first.to(device);
                auto labels = local_batch.second.to(device);
                if (inputs.size(0) == 0) continue;

                auto outputs = model->forward(inputs);
                auto prediction = outputs.argmax(1);
                local_total += labels.sizes()[0];
                local_correct += prediction.eq(labels).sum().item<std::int64_t>();
            }
        }

        auto opts = torch::TensorOptions().dtype(torch::kInt64).device(device);
        const auto local_correct_tensor = torch::tensor(local_correct, opts);
        const auto local_total_tensor = torch::tensor(local_total, opts);
        const auto global_correct_tensor = distdl::sum_scalar_tensor(local_correct_tensor);
        const auto global_total_tensor = distdl::sum_scalar_tensor(local_total_tensor);

        if (rank == 0)
        {
            const auto gc = global_correct_tensor.to(torch::kCPU).item<std::int64_t>();
            const auto gt = global_total_tensor.to(torch::kCPU).item<std::int64_t>();
            const double accuracy = gt > 0 ? (static_cast<double>(gc) / gt * 100.0) : 0.0;
            std::cout << "\nAccuracy on test set: " << accuracy << "%" << std::endl;
        }

        distdl::finalize_distributed_training();
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[Rank " << rank << "] Error: " << ex.what() << std::endl;
        try { distdl::finalize_distributed_training(); } catch (...) {}
        return 1;
    }
}