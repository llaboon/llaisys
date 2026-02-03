#include "op.hpp"
#include "../../utils.hpp"
#include <cmath>

namespace llaisys::ops {

template <typename T>
void rms_norm_kernel(T *out, const T *in, const T *weight, float eps, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        float sum_sq = 0.0f;
        // 1. 计算平方和
        for (size_t j = 0; j < cols; j++) {
            float val = llaisys::utils::cast<float>(in[i * cols + j]);
            sum_sq += val * val;
        }

        // 2. 计算 RMS
        float rms = std::sqrt(sum_sq / cols + eps);
        float inv_rms = 1.0f / rms;

        // 3. 归一化并缩放
        for (size_t j = 0; j < cols; j++) {
            float val = llaisys::utils::cast<float>(in[i * cols + j]);
            float w = llaisys::utils::cast<float>(weight[j]);
            out[i * cols + j] = llaisys::utils::cast<T>(val * inv_rms * w);
        }
    }
}

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // 【修复1】 shape[0] -> shape()[0]
    size_t rows = in->shape()[0];
    size_t cols = in->shape()[1];

    // 【修复2】 dtype -> dtype()
    switch (in->dtype()) {
    case LLAISYS_DTYPE_F32:
        // 【修复3】 data -> data()
        rms_norm_kernel(reinterpret_cast<float*>(out->data()),
                        reinterpret_cast<const float*>(in->data()),
                        reinterpret_cast<const float*>(weight->data()),
                        eps, rows, cols);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_kernel(reinterpret_cast<llaisys::bf16_t*>(out->data()),
                        reinterpret_cast<const llaisys::bf16_t*>(in->data()),
                        reinterpret_cast<const llaisys::bf16_t*>(weight->data()),
                        eps, rows, cols);
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_kernel(reinterpret_cast<llaisys::fp16_t*>(out->data()),
                        reinterpret_cast<const llaisys::fp16_t*>(in->data()),
                        reinterpret_cast<const llaisys::fp16_t*>(weight->data()),
                        eps, rows, cols);
        break;
    default: break;
    }
}
} // namespace llaisys::ops
