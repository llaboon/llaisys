#include "op.hpp"
#include "../../utils.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits> // for -infinity

namespace llaisys::ops {

template <typename T>
void self_attention_kernel(T *attn_val, const T *q, const T *k, const T *v, float scale,
                           size_t seq_len, size_t total_len, 
                           size_t n_heads, size_t n_kv_heads, // 【新增】传入 KV 头数
                           size_t head_dim) {
    
    // 计算 GQA 的组大小 (例如 4个Q头，2个KV头，则每 2 个 Q 共享 1 个 KV)
    size_t group_size = n_heads / n_kv_heads;

    for (size_t s = 0; s < seq_len; s++) { 
        // 计算当前 query 在全局序列中的位置，用于 Causal Mask
        // 假设是推理(Decoding)或预填充(Prefill)，Key通常包含 Past + Current
        // global_pos = (total_len - seq_len) + s
        size_t global_pos = (total_len - seq_len) + s;

        for (size_t h = 0; h < n_heads; h++) { 
            
            // 【GQA 关键修正】映射 Query Head (h) 到 KV Head (kv_h)
            size_t kv_h = h / group_size;

            std::vector<float> scores(total_len);
            float max_score = -1e9f;

            // 1. Q * K^T
            for (size_t t = 0; t < total_len; t++) {
                // 【Causal Mask 修正】如果 t > global_pos，则掩盖 (设为 -inf)
                if (t > global_pos) {
                    scores[t] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                float dot = 0.0f;
                for (size_t d = 0; d < head_dim; d++) {
                    // Q 使用 n_heads 步长
                    float q_val = llaisys::utils::cast<float>(q[(s * n_heads + h) * head_dim + d]);
                    // K 使用 n_kv_heads 步长，并且使用映射后的 kv_h
                    float k_val = llaisys::utils::cast<float>(k[(t * n_kv_heads + kv_h) * head_dim + d]);
                    dot += q_val * k_val;
                }
                scores[t] = dot * scale;
                if (scores[t] > max_score) max_score = scores[t];
            }

            // 2. Softmax
            float sum_exp = 0.0f;
            for (size_t t = 0; t < total_len; t++) {
                if (t > global_pos) {
                    scores[t] = 0.0f; // exp(-inf) = 0
                } else {
                    scores[t] = std::exp(scores[t] - max_score);
                    sum_exp += scores[t];
                }
            }

            // 3. Weighted Sum
            for (size_t d = 0; d < head_dim; d++) {
                float weighted_sum = 0.0f;
                for (size_t t = 0; t < total_len; t++) {
                    if (t > global_pos) continue; // 跳过被 Mask 的部分

                    float prob = scores[t] / sum_exp;
                    // V 使用 n_kv_heads 步长，并使用 kv_h
                    float v_val = llaisys::utils::cast<float>(v[(t * n_kv_heads + kv_h) * head_dim + d]);
                    weighted_sum += prob * v_val;
                }
                attn_val[(s * n_heads + h) * head_dim + d] = llaisys::utils::cast<T>(weighted_sum);
            }
        }
    }
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    size_t seq_len = q->shape()[0];
    size_t n_heads = q->shape()[1];
    size_t head_dim = q->shape()[2];
    
    // 【新增】从 K 的 shape 获取 KV 头数
    size_t total_len = k->shape()[0];
    size_t n_kv_heads = k->shape()[1]; 

    switch (q->dtype()) {
    case LLAISYS_DTYPE_F32:
        self_attention_kernel(reinterpret_cast<float*>(attn_val->data()),
                              reinterpret_cast<const float*>(q->data()),
                              reinterpret_cast<const float*>(k->data()),
                              reinterpret_cast<const float*>(v->data()),
                              scale, seq_len, total_len, n_heads, n_kv_heads, head_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_kernel(reinterpret_cast<llaisys::bf16_t*>(attn_val->data()),
                              reinterpret_cast<const llaisys::bf16_t*>(q->data()),
                              reinterpret_cast<const llaisys::bf16_t*>(k->data()),
                              reinterpret_cast<const llaisys::bf16_t*>(v->data()),
                              scale, seq_len, total_len, n_heads, n_kv_heads, head_dim);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_kernel(reinterpret_cast<llaisys::fp16_t*>(attn_val->data()),
                              reinterpret_cast<const llaisys::fp16_t*>(q->data()),
                              reinterpret_cast<const llaisys::fp16_t*>(k->data()),
                              reinterpret_cast<const llaisys::fp16_t*>(v->data()),
                              scale, seq_len, total_len, n_heads, n_kv_heads, head_dim);
        break;
    default: break;
    }
}
} // namespace llaisys::ops
