#include <torch/torch.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#if defined(USE_C10D_MPI)
#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#endif

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dist_deep_learning_lib.h"

namespace
{
struct DistConfig
{
    int rank = 0;
    int world_size = 1;
    int local_rank = 0;
};

int env_to_int(const char* name, int fallback)
{
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0')
    {
        return fallback;
    }

    try
    {
        return std::stoi(v);
    }
    catch (...)
    {
        return fallback;
    }
}

DistConfig read_dist_config()
{
    DistConfig cfg;

    cfg.rank = env_to_int("RANK", -1);
    if (cfg.rank < 0)
    {
        cfg.rank = env_to_int("OMPI_COMM_WORLD_RANK", -1);
    }
    if (cfg.rank < 0)
    {
        cfg.rank = env_to_int("PMI_RANK", -1);
    }
    if (cfg.rank < 0)
    {
        cfg.rank = env_to_int("MV2_COMM_WORLD_RANK", -1);
    }
    if (cfg.rank < 0)
    {
        cfg.rank = env_to_int("SLURM_PROCID", 0);
    }

    cfg.world_size = env_to_int("WORLD_SIZE", -1);
    if (cfg.world_size < 0)
    {
        cfg.world_size = env_to_int("OMPI_COMM_WORLD_SIZE", -1);
    }
    if (cfg.world_size < 0)
    {
        cfg.world_size = env_to_int("PMI_SIZE", -1);
    }
    if (cfg.world_size < 0)
    {
        cfg.world_size = env_to_int("MV2_COMM_WORLD_SIZE", -1);
    }
    if (cfg.world_size < 0)
    {
        cfg.world_size = env_to_int("SLURM_NTASKS", 1);
    }

    cfg.local_rank = env_to_int("LOCAL_RANK", -1);
    if (cfg.local_rank < 0)
    {
        cfg.local_rank = env_to_int("OMPI_COMM_WORLD_LOCAL_RANK", -1);
    }
    if (cfg.local_rank < 0)
    {
        cfg.local_rank = env_to_int("MV2_COMM_WORLD_LOCAL_RANK", -1);
    }
    if (cfg.local_rank < 0)
    {
        cfg.local_rank = env_to_int("MPI_LOCALRANKID", -1);
    }
    if (cfg.local_rank < 0)
    {
        cfg.local_rank = env_to_int("SLURM_LOCALID", -1);
    }

    if (cfg.local_rank < 0)
    {
        cfg.local_rank = cfg.rank;
    }

    return cfg;
}

c10::intrusive_ptr<c10d::Backend> create_pg_mpi()
{
#if defined(USE_C10D_MPI)
    return c10d::ProcessGroupMPI::createProcessGroupMPI(std::vector<int>{});
#else
    throw std::runtime_error("Libtorch in this environment was built without USE_C10D_MPI");
#endif
}

bool pg_active(const c10::intrusive_ptr<c10d::Backend>& pg, int world_size)
{
    return pg && world_size > 1;
}

void pg_barrier(const c10::intrusive_ptr<c10d::Backend>& pg)
{
    if (!pg)
    {
        return;
    }

    c10d::BarrierOptions opts;
    opts.asyncOp = false;
    pg->barrier(opts)->wait();
}

std::pair<torch::Tensor, torch::Tensor> shard_batch(
    const torch::Tensor& x,
    const torch::Tensor& y,
    int rank,
    int world_size
)
{
    if (world_size <= 1)
    {
        return {x, y};
    }

    const int64_t n = x.size(0);
    const int64_t start = (n * static_cast<int64_t>(rank)) / static_cast<int64_t>(world_size);
    const int64_t end = (n * static_cast<int64_t>(rank + 1)) / static_cast<int64_t>(world_size);

    if (start == end)
    {
        auto xs = x.sizes().vec();
        auto ys = y.sizes().vec();
        xs[0] = 0;
        ys[0] = 0;
        return {
            torch::empty(xs, x.options()),
            torch::empty(ys, y.options())
        };
    }

    return {
        x.slice(0, start, end),
        y.slice(0, start, end)
    };
}

void broadcast_model(torch::nn::Module& module, const c10::intrusive_ptr<c10d::Backend>& pg, int src_rank)
{
    if (!pg_active(pg, pg ? pg->getSize() : 1))
    {
        return;
    }

    c10d::BroadcastOptions opts;
    opts.rootRank = src_rank;
    opts.asyncOp = false;

    torch::NoGradGuard ng;

    for (auto& p : module.parameters())
    {
        if (!p.defined())
        {
            continue;
        }
        torch::Tensor comm = p.detach().to(torch::kCPU).contiguous().clone();
        std::vector<torch::Tensor> t{comm};
        pg->broadcast(t, opts)->wait();
        p.copy_(comm.to(p.device(), p.scalar_type()));
    }

    for (auto& b : module.buffers())
    {
        if (!b.defined())
        {
            continue;
        }
        torch::Tensor comm = b.detach().to(torch::kCPU).contiguous().clone();
        std::vector<torch::Tensor> t{comm};
        pg->broadcast(t, opts)->wait();
        b.copy_(comm.to(b.device(), b.scalar_type()));
    }
}

void sync_grads_allreduce(
    torch::nn::Module& module,
    const c10::intrusive_ptr<c10d::Backend>& pg,
    int world_size
)
{
    if (!pg_active(pg, world_size))
    {
        return;
    }

    c10d::AllreduceOptions ao;
    ao.reduceOp = c10d::ReduceOp::SUM;
    ao.asyncOp = false;

    torch::NoGradGuard ng;

    for (auto& p : module.parameters())
    {
        auto g = p.grad();
        if (!g.defined())
        {
            continue;
        }

        torch::Tensor comm = g.detach().to(torch::kCPU).contiguous().clone();
        std::vector<torch::Tensor> t{comm};

        pg->allreduce(t, ao)->wait();
        comm.div_(static_cast<double>(world_size));

        g.copy_(comm.to(g.device(), g.scalar_type()));
    }
}

torch::Tensor average_loss_allreduce(
    const torch::Tensor& local_loss,
    const c10::intrusive_ptr<c10d::Backend>& pg,
    int world_size
)
{
    if (!pg_active(pg, world_size))
    {
        return local_loss.detach().clone();
    }

    c10d::AllreduceOptions ao;
    ao.reduceOp = c10d::ReduceOp::SUM;
    ao.asyncOp = false;

    torch::Tensor comm = local_loss.detach();
    if (comm.is_cuda())
    {
        comm = comm.to(torch::kCPU);
    }
    comm = comm.clone();

    std::vector<torch::Tensor> t{comm};
    pg->allreduce(t, ao)->wait();
    comm.div_(static_cast<double>(world_size));

    return comm.to(local_loss.device(), local_loss.scalar_type());
}

torch::Tensor sum_i64_allreduce(
    std::int64_t local_value,
    const c10::intrusive_ptr<c10d::Backend>& pg,
    const torch::Device& device,
    int world_size
)
{
    torch::Tensor local = torch::tensor(local_value, torch::TensorOptions().dtype(torch::kInt64).device(device));
    if (!pg_active(pg, world_size))
    {
        return local;
    }

    c10d::AllreduceOptions ao;
    ao.reduceOp = c10d::ReduceOp::SUM;
    ao.asyncOp = false;

    torch::Tensor comm = local.is_cuda() ? local.to(torch::kCPU) : local;
    comm = comm.clone();
    std::vector<torch::Tensor> t{comm};

    pg->allreduce(t, ao)->wait();

    return comm.to(device, torch::kInt64);
}
} // namespace

int main()
{
    DistConfig cfg = read_dist_config();
    c10::intrusive_ptr<c10d::Backend> pg;

    try
    {
        pg = create_pg_mpi();
        cfg.rank = pg->getRank();
        cfg.world_size = pg->getSize();
        if (cfg.local_rank < 0)
        {
            cfg.local_rank = cfg.rank;
        }

        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available())
        {
            const int gpu_count = torch::cuda::device_count();
            const int gpu_index = gpu_count > 0 ? (cfg.local_rank % gpu_count) : 0;
            device = torch::Device(torch::kCUDA, gpu_index);
        }

        if (cfg.rank == 0)
        {
            std::cout << "[torch.dist/c10d] backend=" << pg->getBackendName()
                      << " world_size=" << cfg.world_size << std::endl;
        }

        torch::manual_seed(1337);

        const std::string data_path = "./data/cifar-10-batches-bin";

        auto train_dataset = CIFAR10(data_path, CIFAR10::Mode::kTrain)
            .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
            .map(torch::data::transforms::Stack<>());

        auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset),
            torch::data::DataLoaderOptions().batch_size(128).drop_last(true)
        );

        auto test_dataset = CIFAR10(data_path, CIFAR10::Mode::kTest)
            .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
            .map(torch::data::transforms::Stack<>());

        auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(test_dataset),
            torch::data::DataLoaderOptions().batch_size(128)
        );

        distdl::ResNet model(10);
        model->to(device);
        broadcast_model(*model, pg, 0);

        torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.001));

        const int epochs = 1;
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            model->train();
            double running_loss = 0.0;
            int batch_idx = 0;

            torch::manual_seed(1337 + epoch);

            for (auto& batch : *train_loader)
            {
                auto local = shard_batch(batch.data, batch.target, cfg.rank, cfg.world_size);
                auto inputs = local.first.to(device);
                auto labels = local.second.to(device);
                if (inputs.size(0) == 0)
                {
                    continue;
                }

                optimizer.zero_grad();
                auto outputs = model->forward(inputs);
                auto loss = torch::nn::functional::cross_entropy(outputs, labels);
                loss.backward();

                sync_grads_allreduce(*model, pg, cfg.world_size);
                optimizer.step();

                auto avg_loss = average_loss_allreduce(loss.detach(), pg, cfg.world_size);
                running_loss += avg_loss.item<double>();

                if (cfg.rank == 0 && (batch_idx + 1) % 10 == 0)
                {
                    std::cout << "[Epoch " << (epoch + 1) << ", Batch " << (batch_idx + 1)
                              << "] Loss: " << (running_loss / 10.0) << std::endl;
                    running_loss = 0.0;
                }
                ++batch_idx;
            }
        }

        pg_barrier(pg);

        model->eval();
        std::int64_t local_correct = 0;
        std::int64_t local_total = 0;

        {
            torch::NoGradGuard no_grad;
            for (const auto& batch : *test_loader)
            {
                auto local = shard_batch(batch.data, batch.target, cfg.rank, cfg.world_size);
                auto inputs = local.first.to(device);
                auto labels = local.second.to(device);
                if (inputs.size(0) == 0)
                {
                    continue;
                }

                auto outputs = model->forward(inputs);
                auto prediction = outputs.argmax(1);
                local_total += labels.sizes()[0];
                local_correct += prediction.eq(labels).sum().item<std::int64_t>();
            }
        }

        auto global_correct = sum_i64_allreduce(local_correct, pg, device, cfg.world_size).item<std::int64_t>();
        auto global_total = sum_i64_allreduce(local_total, pg, device, cfg.world_size).item<std::int64_t>();

        if (cfg.rank == 0)
        {
            const double accuracy = global_total > 0
                ? (static_cast<double>(global_correct) / static_cast<double>(global_total) * 100.0)
                : 0.0;
            std::cout << "Accuracy: " << accuracy << "%" << std::endl;
        }

        pg_barrier(pg);
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[Rank " << cfg.rank << "] Error: " << ex.what() << std::endl;
        try
        {
            pg_barrier(pg);
        }
        catch (...)
        {
        }
        return 1;
    }
}
