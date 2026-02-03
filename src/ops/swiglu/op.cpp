#include "op.hpp"
#include "../../utils.hpp"
#include <cmath>

namespace llaisys::ops {

template <typename T>
void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        float g = llaisys::utils::cast<float>(gate[i]);
        float u = llaisys::utils::cast<float>(up[i]);

        // SiLU = x / (1 + exp(-x))
        float silu = g / (1.0f + std::exp(-g));

        out[i] = llaisys::utils::cast<T>(u * silu);
    }
}

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    size_t numel = out->numel();

    // 【修复】 dtype -> dtype()
    switch (gate->dtype()) {
    case LLAISYS_DTYPE_F32:
        // 【修复】 data -> data()
        swiglu_kernel(reinterpret_cast<float*>(out->data()),
                      reinterpret_cast<const float*>(gate->data()),
                      reinterpret_cast<const float*>(up->data()), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_kernel(reinterpret_cast<llaisys::bf16_t*>(out->data()),
                      reinterpret_cast<const llaisys::bf16_t*>(gate->data()),
                      reinterpret_cast<const llaisys::bf16_t*>(up->data()), numel);
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_kernel(reinterpret_cast<llaisys::fp16_t*>(out->data()),
                      reinterpret_cast<const llaisys::fp16_t*>(gate->data()),
                      reinterpret_cast<const llaisys::fp16_t*>(up->data()), numel);
        break;
    default: break;
    }
}
} // namespace llaisys::ops
