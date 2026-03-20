#include <torch/torch.h>
#include <vector>
#include <cstddef>
#include <cstring>
#include <stdexcept>

template <typename T>
void appendToBuffer(std::vector<std::byte>& buffer, const T& value)
{
    const std::byte* ptr = reinterpret_cast<const std::byte*>(&value);
    buffer.insert(buffer.end(), ptr, ptr + sizeof(T));
}

template <typename T>
T readFromBuffer(const std::vector<std::byte>& buffer, size_t& offset)
{
    if (offset + sizeof(T) > buffer.size())
    {
        throw std::runtime_error("Read error: index out of buffer bounds");
    }
    T value;
    std::memcpy(&value, buffer.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

std::vector<std::byte> serialize(torch::Tensor tensor)
{
    tensor = tensor.cpu().contiguous();

    int64_t nDim = tensor.dim();
    size_t dataBytes = tensor.numel() * tensor.element_size();

    size_t totalBytes = sizeof(int8_t)
                      + sizeof(int64_t)
                      + (nDim * sizeof(int64_t))
                      + dataBytes;

    std::vector<std::byte> buffer;
    buffer.reserve(totalBytes);

    int8_t scalarType = static_cast<int8_t>(tensor.scalar_type());
    appendToBuffer(buffer, scalarType);

    appendToBuffer(buffer, nDim);

    c10::IntArrayRef sizes = tensor.sizes();
    for (int64_t i = 0; i < nDim; ++i)
    {
        appendToBuffer(buffer, static_cast<int64_t>(sizes[i]));
    }

    const std::byte* dataPtr = reinterpret_cast<const std::byte*>(tensor.data_ptr());
    buffer.insert(buffer.end(), dataPtr, dataPtr + dataBytes);

    return buffer;
}

torch::Tensor deserialize(const std::vector<std::byte>& buffer)
{
    size_t offset = 0;

    int8_t typeVal = readFromBuffer<int8_t>(buffer, offset);
    torch::ScalarType scalarType = static_cast<torch::ScalarType>(typeVal);

    int64_t nDim = readFromBuffer<int64_t>(buffer, offset);

    std::vector<int64_t> shape(nDim);
    for (int64_t i = 0; i < nDim; ++i)
    {
        shape[i] = readFromBuffer<int64_t>(buffer, offset);
    }

    torch::Tensor tensor = torch::empty(shape, torch::TensorOptions().dtype(scalarType).device(torch::kCPU));

    size_t dataBytes = tensor.numel() * tensor.element_size();
    if (offset + dataBytes > buffer.size())
    {
        throw std::runtime_error("Deserialization error: not enough data of payload");
    }
    std::memcpy(tensor.data_ptr(), buffer.data() + offset, dataBytes);

    return tensor;
}