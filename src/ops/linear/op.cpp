#include "op.hpp"
#include "../../utils.hpp"

namespace llaisys::ops {

template <typename T>
void linear_kernel(T *out, const T *in, const T *weight, const T *bias,
                   size_t M, size_t N, size_t K) {
    // M: input rows (batch * seq)
    // N: output features (weight rows)
    // K: input features (weight cols)

    // 简单的朴素矩阵乘法
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                float a = llaisys::utils::cast<float>(in[m * K + k]);
                // Weight is [N, K], so we access row n, col k
                float b = llaisys::utils::cast<float>(weight[n * K + k]);
                sum += a * b;
            }
            if (bias) {
                sum += llaisys::utils::cast<float>(bias[n]);
            }
            out[m * N + n] = llaisys::utils::cast<T>(sum);
        }
    }
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // 假设输入是 [M, K], 权重是 [N, K]
    size_t M = in->shape()[0];
    size_t K = in->shape()[1];
    size_t N = weight->shape()[0];

    auto dtype = in->dtype();

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        linear_kernel(reinterpret_cast<float*>(out->data()),
                      reinterpret_cast<const float*>(in->data()),
                      reinterpret_cast<const float*>(weight->data()),
                      bias ? reinterpret_cast<const float*>(bias->data()) : nullptr,
                      M, N, K);
        break;
    case LLAISYS_DTYPE_BF16:
        linear_kernel(reinterpret_cast<llaisys::bf16_t*>(out->data()),
                      reinterpret_cast<const llaisys::bf16_t*>(in->data()),
                      reinterpret_cast<const llaisys::bf16_t*>(weight->data()),
                      bias ? reinterpret_cast<const llaisys::bf16_t*>(bias->data()) : nullptr,
                      M, N, K);
        break;
    case LLAISYS_DTYPE_F16:
        linear_kernel(reinterpret_cast<llaisys::fp16_t*>(out->data()),
                      reinterpret_cast<const llaisys::fp16_t*>(in->data()),
                      reinterpret_cast<const llaisys::fp16_t*>(weight->data()),
                      bias ? reinterpret_cast<const llaisys::fp16_t*>(bias->data()) : nullptr,
                      M, N, K);
        break;
    default: break;
    }
}
} // namespace llaisys::ops
