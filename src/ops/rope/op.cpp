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
                // 1. 频率计算维持 double
                double freq_expon = -2.0 * static_cast<double>(j) / static_cast<double>(head_dim);
                double freq = static_cast<double>(pos) * std::pow(static_cast<double>(theta), freq_expon);
                
                // 2. 三角函数维持 double
                double cos_val = std::cos(freq);
                double sin_val = std::sin(freq);

                // 3. 【关键修改】读取输入也转为 double
                double x = static_cast<double>(llaisys::utils::cast<float>(in[offset + j]));
                double y = static_cast<double>(llaisys::utils::cast<float>(in[offset + j + half_dim]));

                // 4. 【关键修改】旋转运算全程使用 double，最大程度减少中间误差
                double out_x = x * cos_val - y * sin_val;
                double out_y = x * sin_val + y * cos_val;

                // 5. 最后再转回目标类型 T
                out[offset + j] = llaisys::utils::cast<T>(static_cast<float>(out_x));
                out[offset + j + half_dim] = llaisys::utils::cast<T>(static_cast<float>(out_y));
            }
        }
    }
}

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    size_t seq_len = in->shape()[0];
    size_t n_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];

    const int64_t* pos_ptr = reinterpret_cast<const int64_t*>(pos_ids->data());

    switch (in->dtype()) {
    case LLAISYS_DTYPE_F32:
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
