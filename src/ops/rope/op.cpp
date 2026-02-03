#include "op.hpp"
#include "../../utils.hpp"
#include <cmath>

namespace llaisys::ops {

template <typename T>
void rope_kernel(T *out, const T *in, const int64_t *pos_ids, float theta,
                 size_t seq_len, size_t n_heads, size_t head_dim) {

    for (size_t s = 0; s < seq_len; s++) {
        int64_t pos = pos_ids[s];
        for (size_t h = 0; h < n_heads; h++) {
            size_t offset = s * n_heads * head_dim + h * head_dim;
            size_t half_dim = head_dim / 2;

            for (size_t j = 0; j < half_dim; j++) {
                float freq = static_cast<float>(pos) * std::pow(theta, -2.0f * j / head_dim);
                float cos_val = std::cos(freq);
                float sin_val = std::sin(freq);

                float x = llaisys::utils::cast<float>(in[offset + j]);
                float y = llaisys::utils::cast<float>(in[offset + j + half_dim]);

                float out_x = x * cos_val - y * sin_val;
                float out_y = x * sin_val + y * cos_val;

                out[offset + j] = llaisys::utils::cast<T>(out_x);
                out[offset + j + half_dim] = llaisys::utils::cast<T>(out_y);
            }
        }
    }
}

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // 【修复】 shape -> shape()
    size_t seq_len = in->shape()[0];
    size_t n_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];

    // 【修复】 pos_ids->data -> pos_ids->data()
    const int64_t* pos_ptr = reinterpret_cast<const int64_t*>(pos_ids->data());

    // 【修复】 dtype -> dtype()
    switch (in->dtype()) {
    case LLAISYS_DTYPE_F32:
        // 【修复】 data -> data()
        rope_kernel(reinterpret_cast<float*>(out->data()),
                    reinterpret_cast<const float*>(in->data()),
                    pos_ptr, theta, seq_len, n_heads, head_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_kernel(reinterpret_cast<llaisys::bf16_t*>(out->data()),
                    reinterpret_cast<const llaisys::bf16_t*>(in->data()),
                    pos_ptr, theta, seq_len, n_heads, head_dim);
        break;
    case LLAISYS_DTYPE_F16:
        rope_kernel(reinterpret_cast<llaisys::fp16_t*>(out->data()),
                    reinterpret_cast<const llaisys::fp16_t*>(in->data()),
                    pos_ptr, theta, seq_len, n_heads, head_dim);
        break;
    default: break;
    }
}
} // namespace llaisys::ops
