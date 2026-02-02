#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>
#include <stdexcept> // 引入异常处理

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

// ============================================
// 任务 1.1: Load
// ============================================
void Tensor::load(const void *src_) {
    // 确保切换到正确的设备上下文
    core::context().setDevice(this->deviceType(), this->deviceId());

    // 调用 Runtime API 执行从 Host 到 Device 的内存拷贝
    core::context().runtime().api()->memcpy_sync(
        this->data(),                       // 目标地址 (Storage指针 + Offset)
        src_,                               // 源地址 (Host数据)
        this->numel() * this->elementSize(), // 拷贝字节数
        LLAISYS_MEMCPY_H2D                  // 拷贝方向：Host To Device
    );
}

// ============================================
// 任务 1.2: isContiguous
// ============================================
bool Tensor::isContiguous() const {
    size_t z = 1;
    // 从最后一个维度向前检查
    // 连续张量要求：stride[i] == stride[i+1] * shape[i+1]
    // 且最后一个维度的 stride 必须为 1
    for (int i = _meta.shape.size() - 1; i >= 0; --i) {
        if (_meta.strides[i] != static_cast<ptrdiff_t>(z)) {
            return false;
        }
        z *= _meta.shape[i];
    }
    return true;
}

// ============================================
// 任务 1.4: Permute
// ============================================
tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    if (order.size() != _meta.shape.size()) {
        throw std::runtime_error("Permute order size mismatch");
    }

    TensorMeta new_meta;
    new_meta.dtype = _meta.dtype;
    new_meta.shape.reserve(order.size());
    new_meta.strides.reserve(order.size());

    for (size_t i : order) {
        if (i >= _meta.shape.size()) {
             throw std::runtime_error("Permute index out of bounds");
        }
        new_meta.shape.push_back(_meta.shape[i]);
        new_meta.strides.push_back(_meta.strides[i]);
    }

    // 返回新张量，共享 Storage，Offset 不变
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

// ============================================
// 任务 1.3: View
// ============================================
tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    size_t new_numel = 1;
    for (auto s : shape) new_numel *= s;

    if (new_numel != this->numel()) {
        throw std::runtime_error("View shape mismatch: total element count must be the same");
    }

    // 关键检查：非连续张量无法直接 View，必须先 Contiguous
    if (!this->isContiguous()) {
        throw std::runtime_error("View can only be called on contiguous tensors");
    }

    // 重新计算连续的 strides
    std::vector<ptrdiff_t> new_strides(shape.size());
    size_t z = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        new_strides[i] = static_cast<ptrdiff_t>(z);
        z *= shape[i];
    }

    TensorMeta new_meta{_meta.dtype, shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

// ============================================
// 任务 1.5: Slice
// ============================================
tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    if (dim >= _meta.shape.size()) {
        throw std::runtime_error("Slice dimension out of range");
    }
    if (start > end || end > _meta.shape[dim]) {
        throw std::runtime_error("Invalid slice range");
    }

    TensorMeta new_meta = _meta;
    // 修改切片维度的 shape
    new_meta.shape[dim] = end - start;

    // 修改 offset：偏移量增加 = 起始索引 * 该维度的 stride * 每个元素的大小
    size_t offset_add = start * _meta.strides[dim] * elementSize();
    size_t new_offset = _offset + offset_add;

    // Strides 保持不变
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}

// ============================================
// 挑战任务 (本次作业可能不测试，但为了防止崩溃，提供基础抛出异常)
// ============================================

tensor_t Tensor::contiguous() const {
    // 如果已经是连续的，直接返回自己
    if (this->isContiguous()) {
        // 使用拷贝构造或创建一个指向同一对象的新智能指针
        // 为了安全起见，这里简单抛出未实现，除非你有深拷贝的实现
        // 实际上作业1.1-1.6不包含此任务，留空或抛异常即可
        throw std::runtime_error("Function 'contiguous' is not implemented yet.");
    }
    throw std::runtime_error("Function 'contiguous' is not implemented yet.");
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    // 如果是连续的，可以直接复用 view
    if (this->isContiguous()) {
        return this->view(shape);
    }
    throw std::runtime_error("Function 'reshape' on non-contiguous tensor is not implemented yet.");
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    throw std::runtime_error("Function 'to' (device transfer) is not implemented yet.");
}

} // namespace llaisys


} // namespace llaisys
