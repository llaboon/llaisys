#include "op.hpp"
#include "../../utils.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace llaisys::ops {

template <typename T>
void self_attention_kernel(T *attn_val, const T *q, const T *k, const T *v, float scale,
                           size_t seq_len, size_t total_len, size_t n_heads, size_t head_dim) {

    for (size_t s = 0; s < seq_len; s++) { // Query 序列
        for (size_t h = 0; h < n_heads; h++) { // Head

            std::vector<float> scores(total_len);
            float max_score = -1e9f;

            // 1. Q * K^T
            for (size_t t = 0; t < total_len; t++) {
                float dot = 0.0f;
                for (size_t d = 0; d < head_dim; d++) {
                    float q_val = llaisys::utils::cast<float>(q[(s * n_heads + h) * head_dim + d]);
                    float k_val = llaisys::utils::cast<float>(k[(t * n_heads + h) * head_dim + d]);
                    dot += q_val * k_val;
                }
                scores[t] = dot * scale;
                if (scores[t] > max_score) max_score = scores[t];
            }

            // 2. Softmax
            float sum_exp = 0.0f;
            for (size_t t = 0; t < total_len; t++) {
                scores[t] = std::exp(scores[t] - max_score);
                sum_exp += scores[t];
            }

            // 3. Weighted Sum
            for (size_t d = 0; d < head_dim; d++) {
                float weighted_sum = 0.0f;
                for (size_t t = 0; t < total_len; t++) {
                    float prob = scores[t] / sum_exp;
                    float v_val = llaisys::utils::cast<float>(v[(t * n_heads + h) * head_dim + d]);
                    weighted_sum += prob * v_val;
                }
                attn_val[(s * n_heads + h) * head_dim + d] = llaisys::utils::cast<T>(weighted_sum);
            }
        }
    }
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // 【修复】 shape -> shape()
    size_t seq_len = q->shape()[0];
    size_t n_heads = q->shape()[1];
    size_t head_dim = q->shape()[2];
    size_t total_len = k->shape()[0];

    // 【修复】 dtype -> dtype()
    switch (q->dtype()) {
    case LLAISYS_DTYPE_F32:
        // 【修复】 data -> data()
        self_attention_kernel(reinterpret_cast<float*>(attn_val->data()),
                              reinterpret_cast<const float*>(q->data()),
                              reinterpret_cast<const float*>(k->data()),
                              reinterpret_cast<const float*>(v->data()),
                              scale, seq_len, total_len, n_heads, head_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_kernel(reinterpret_cast<llaisys::bf16_t*>(attn_val->data()),
                              reinterpret_cast<const llaisys::bf16_t*>(q->data()),
                              reinterpret_cast<const llaisys::bf16_t*>(k->data()),
                              reinterpret_cast<const llaisys::bf16_t*>(v->data()),
                              scale, seq_len, total_len, n_heads, head_dim);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_kernel(reinterpret_cast<llaisys::fp16_t*>(attn_val->data()),
                              reinterpret_cast<const llaisys::fp16_t*>(q->data()),
                              reinterpret_cast<const llaisys::fp16_t*>(k->data()),
                              reinterpret_cast<const llaisys::fp16_t*>(v->data()),
                              scale, seq_len, total_len, n_heads, head_dim);
        break;
    default: break;
    }
}
} // namespace llaisys::ops
