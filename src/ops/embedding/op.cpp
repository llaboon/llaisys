#include "op.hpp"
#include <cstring>

namespace llaisys::ops {

template <typename T>
void embedding_kernel(T *out, const int64_t *indices, const T *weight,
                      size_t num_indices, size_t embedding_dim) {
    for (size_t i = 0; i < num_indices; i++) {
        int64_t idx = indices[i];
        const T* src = weight + idx * embedding_dim;
        T* dst = out + i * embedding_dim;
        // 直接内存拷贝
        std::memcpy(dst, src, embedding_dim * sizeof(T));
    }
}

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // 【修复1】 index->numel() 本身就是函数调用，这行是对的
    size_t num_indices = index->numel();

    // 【修复2】 weight->shape[1] -> weight->shape()[1]
    size_t embedding_dim = weight->shape()[1];

    // 【修复3】 weight->dtype -> weight->dtype()
    auto dtype = weight->dtype();

    // 【修复4】 index->data -> index->data()
    const int64_t* indices_ptr = reinterpret_cast<const int64_t*>(index->data());

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        // 【修复5】 data -> data()
        embedding_kernel(reinterpret_cast<float*>(out->data()), indices_ptr,
                         reinterpret_cast<const float*>(weight->data()), num_indices, embedding_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_kernel(reinterpret_cast<llaisys::bf16_t*>(out->data()), indices_ptr,
                         reinterpret_cast<const llaisys::bf16_t*>(weight->data()), num_indices, embedding_dim);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_kernel(reinterpret_cast<llaisys::fp16_t*>(out->data()), indices_ptr,
                         reinterpret_cast<const llaisys::fp16_t*>(weight->data()), num_indices, embedding_dim);
        break;
    default: break;
    }
}
} // namespace llaisys::ops
