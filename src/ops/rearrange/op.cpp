#include "op.hpp"
#include "../../utils.hpp"
#include <cstring>
namespace llaisys::ops {

// 这是一个简化的实现，假设只是进行数据搬运。
// 真正的 rearrange 需要根据 input 的 shape 和 stride 计算偏移量。
// 这里假设 tensor_t 包含 strides 信息。如果 tensor_t 没有 stride，这个函数可能只是 reshape (无操作) 或 纯拷贝。

template <typename T>
void rearrange_kernel(T *out, const T *in,
                      const std::vector<int64_t>& shape,
                      const std::vector<int64_t>& in_strides) {
    // 示例：处理 2D 情况，多维需要递归
    if (shape.size() == 2) {
        size_t dim0 = shape[0];
        size_t dim1 = shape[1];
        size_t stride0 = in_strides[0];
        size_t stride1 = in_strides[1];

        for (size_t i = 0; i < dim0; ++i) {
            for (size_t j = 0; j < dim1; ++j) {
                // Input offset based on strides
                size_t in_idx = i * stride0 + j * stride1;
                // Output is assumed contiguous (Row-Major)
                size_t out_idx = i * dim1 + j;

                out[out_idx] = in[in_idx];
            }
        }
    } else {
        // Fallback or recursive implementation needed for N-dims
        // 这里的代码取决于 tensor_t 的具体实现细节
    }
}

void rearrange(tensor_t out, tensor_t in) {
    // 如果没有 stride 信息，或者只是想做一次拷贝
    if (in->dtype() == LLAISYS_DTYPE_F32) {
         // 这里的实现高度依赖 tensor_t 是否暴露 stride
         // 如果没有暴露 stride，仅作 memcpy
         size_t num_bytes = in->numel() * sizeof(float);
         std::memcpy(out->data(), in->data(), num_bytes);
    }
    // 添加其他类型的支持...
}
} // namespace llaisys::ops
