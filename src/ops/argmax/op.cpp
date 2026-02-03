#include "op.hpp"
#include "../../utils.hpp"
#include <cmath>
#include <limits>

namespace llaisys::ops {

template <typename T>
void argmax_kernel(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    // 初始化为负无穷
    float current_max = -std::numeric_limits<float>::infinity();
    int64_t current_idx = 0;

    for (size_t i = 0; i < numel; i++) {
        // 统一转为 float 进行比较
        float val = llaisys::utils::cast<float>(vals[i]);
        if (val > current_max) {
            current_max = val;
            current_idx = i;
        }
    }

    *max_idx = current_idx;
    *max_val = llaisys::utils::cast<T>(current_max);
}

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    size_t numel = vals->numel();

    // 【修复 1】 vals->dtype -> vals->dtype()
    auto dtype = vals->dtype();

    // 【修复 2】 max_idx->data -> max_idx->data()
    int64_t* idx_ptr = reinterpret_cast<int64_t*>(max_idx->data());

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        // 【修复 3】 max_val->data(), vals->data()
        argmax_kernel(idx_ptr, reinterpret_cast<float*>(max_val->data()),
                      reinterpret_cast<const float*>(vals->data()), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_kernel(idx_ptr, reinterpret_cast<llaisys::bf16_t*>(max_val->data()),
                      reinterpret_cast<const llaisys::bf16_t*>(vals->data()), numel);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_kernel(idx_ptr, reinterpret_cast<llaisys::fp16_t*>(max_val->data()),
                      reinterpret_cast<const llaisys::fp16_t*>(vals->data()), numel);
        break;
    default:
        break;
    }
}
} // namespace llaisys::ops
