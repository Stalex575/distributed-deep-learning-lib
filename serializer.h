#pragma once

#include <torch/torch.h>
#include <vector>
#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace distdl::detail
{
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
} // namespace distdl::detail

namespace distdl
{
inline bool is_supported_scalar_type(torch::ScalarType st)
{
    switch (st)
    {
    case torch::kFloat32:
    case torch::kFloat64:
    case torch::kInt32:
    case torch::kInt64:
    case torch::kUInt8:
    case torch::kInt8:
    case torch::kInt16:
    case torch::kFloat16:
    case torch::kBFloat16:
    case torch::kBool:
        return true;
    default:
        return false;
    }
}

inline std::vector<std::byte> serialize(torch::Tensor tensor)
{
    tensor = tensor.cpu().contiguous();

    if (!is_supported_scalar_type(tensor.scalar_type()))
    {
        throw std::runtime_error(
            "serialize: unsupported scalar type "
            + std::string(toString(tensor.scalar_type()))
        );
    }

    int64_t nDim = tensor.dim();
    size_t dataBytes = tensor.numel() * tensor.element_size();

    size_t totalBytes = sizeof(int8_t)
                      + sizeof(int64_t)
                      + (nDim * sizeof(int64_t))
                      + dataBytes;

    std::vector<std::byte> buffer;
    buffer.reserve(totalBytes);

    int8_t scalarType = static_cast<int8_t>(tensor.scalar_type());
    detail::appendToBuffer(buffer, scalarType);

    detail::appendToBuffer(buffer, nDim);

    c10::IntArrayRef sizes = tensor.sizes();
    for (int64_t i = 0; i < nDim; ++i)
    {
        detail::appendToBuffer(buffer, static_cast<int64_t>(sizes[i]));
    }

    const std::byte* dataPtr = reinterpret_cast<const std::byte*>(tensor.data_ptr());
    buffer.insert(buffer.end(), dataPtr, dataPtr + dataBytes);

    return buffer;
}

inline torch::Tensor deserialize(const std::vector<std::byte>& buffer)
{
    size_t offset = 0;

    int8_t typeVal = detail::readFromBuffer<int8_t>(buffer, offset);
    torch::ScalarType scalarType = static_cast<torch::ScalarType>(typeVal);

    if (!is_supported_scalar_type(scalarType))
    {
        throw std::runtime_error(
            "deserialize: unsupported or corrupt scalar type value "
            + std::to_string(static_cast<int>(typeVal))
        );
    }

    int64_t nDim = detail::readFromBuffer<int64_t>(buffer, offset);

    if (nDim < 0 || nDim > 32)
    {
        throw std::runtime_error(
            "deserialize: suspicious ndim=" + std::to_string(nDim)
        );
    }

    std::vector<int64_t> shape(nDim);
    for (int64_t i = 0; i < nDim; ++i)
    {
        shape[i] = detail::readFromBuffer<int64_t>(buffer, offset);
    }

    torch::Tensor tensor = torch::empty(shape, torch::TensorOptions().dtype(scalarType).device(torch::kCPU));

    size_t dataBytes = tensor.numel() * tensor.element_size();
    if (offset + dataBytes > buffer.size())
    {
        throw std::runtime_error("Deserialization error: not enough data in payload");
    }
    std::memcpy(tensor.data_ptr(), buffer.data() + offset, dataBytes);

    return tensor;
}
} // namespace distdl
