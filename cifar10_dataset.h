#pragma once

#include <torch/torch.h>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>

class CIFAR10 : public torch::data::datasets::Dataset<CIFAR10>
{
public:
    enum class Mode
    {
        kTrain,
        kTest
    };

private:
    torch::Tensor images_;
    torch::Tensor targets_;

    void read_batch(const std::string& path, std::vector<uint8_t>& image_buffer, std::vector<uint8_t>& label_buffer)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("Cannot open file: " + path + ". Make sure you downloaded the binary CIFAR-10 dataset.");
        }

        const int num_images = 10000;
        const int image_size = 3072;
        
        for (int i = 0; i < num_images; ++i)
        {
            uint8_t label;
            file.read(reinterpret_cast<char*>(&label), 1);
            label_buffer.push_back(label);

            std::vector<uint8_t> image(image_size);
            file.read(reinterpret_cast<char*>(image.data()), image_size);
            image_buffer.insert(image_buffer.end(), image.begin(), image.end());
        }
    }

public:
    CIFAR10(const std::string& root, Mode mode = Mode::kTrain)
    {
        std::vector<uint8_t> image_buffer;
        std::vector<uint8_t> label_buffer;
        int num_images = 0;

        if (mode == Mode::kTrain)
        {
            std::cout << "Loading CIFAR-10 Training dataset..." << std::endl;
            for (int i = 1; i <= 5; ++i)
            {
                read_batch(root + "/data_batch_" + std::to_string(i) + ".bin", image_buffer, label_buffer);
            }
            num_images = 50000;
        }
        else
        {
            std::cout << "Loading CIFAR-10 Test dataset..." << std::endl;
            read_batch(root + "/test_batch.bin", image_buffer, label_buffer);
            num_images = 10000;
        }

        auto images_tensor = torch::from_blob(image_buffer.data(), {num_images, 3, 32, 32}, torch::kByte).clone();
        auto labels_tensor = torch::from_blob(label_buffer.data(), {num_images}, torch::kByte).clone();

        images_ = images_tensor.to(torch::kFloat32).div_(255.0);
        targets_ = labels_tensor.to(torch::kInt64);
    }

    torch::data::Example<> get(size_t index) override
    {
        return {images_[index], targets_[index]};
    }

    torch::optional<size_t> size() const override
    {
        return images_.size(0);
    }
};
