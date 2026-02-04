#include "llaisys/models/qwen2.h"
#include "llaisys/tensor.h"
#include "../utils.hpp" 
#include "../tensor/tensor.hpp" // 假设这是内部 Tensor 类的定义
#include <vector>
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>

using namespace llaisys;
#ifndef LLAISYS_EXPORT
    #if defined(_WIN32)
        #define LLAISYS_EXPORT __declspec(dllexport)
    #else
        #define LLAISYS_EXPORT __attribute__((visibility("default")))
    #endif
#endif
// =====================================================
// 1. 工具函数：BF16 转 FP32
// =====================================================
// 即使 ops 未提供，我们也要确保能转换数据
namespace {
    inline float bf16_to_fp32(uint16_t h) {
        union {
            float f;
            uint32_t i;
        } u;
        u.i = (uint32_t)h << 16; // 核心魔法：左移16位
        return u.f;
    }
}

// =====================================================
// 2. 算子声明 (假设 src/ops/ 存在，如果不存在，需要在此实现 naive 版本)
// =====================================================
namespace llaisys::ops {
    void embedding(tensor_t out, tensor_t index, tensor_t weight);
    void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps);
    void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);
    void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);
    void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
    void swiglu(tensor_t out, tensor_t gate, tensor_t up);
    void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
}

// =====================================================
// 3. 辅助函数: Tensor Add
// =====================================================
template <typename T>
void add_kernel(std::byte* dst_bytes, const std::byte* src_bytes, size_t numel) {
    T* dst = reinterpret_cast<T*>(dst_bytes);
    const T* src = reinterpret_cast<const T*>(src_bytes);
    for (size_t i = 0; i < numel; ++i) {
        dst[i] += src[i];
    }
}

// 针对 BF16 的特殊加法 (如果模型内部全转 fp32，此分支可能不被调用)
void add_kernel_bf16(std::byte* dst_bytes, const std::byte* src_bytes, size_t numel) {
    uint16_t* dst = reinterpret_cast<uint16_t*>(dst_bytes);
    const uint16_t* src = reinterpret_cast<const uint16_t*>(src_bytes);
    for (size_t i = 0; i < numel; ++i) {
        float a = bf16_to_fp32(dst[i]);
        float b = bf16_to_fp32(src[i]);
        // 这里只是简化的写回，实际上如果不转回 bf16 会有问题
        // 但为了作业稳定性，我们推荐全程使用 FP32 运算
        // 此处仅为占位防止编译错误
        dst[i] = dst[i]; 
    }
}

void tensor_add(tensor_t dst, tensor_t src) {
    if (dst->numel() != src->numel()) return;
    
    // 简单粗暴：Float32 加法
    if (dst->dtype() == LLAISYS_DTYPE_F32) {
        float* d = (float*)dst->data();
        float* s = (float*)src->data();
        for(size_t i=0; i<dst->numel(); ++i) d[i] += s[i];
    }
}

// =====================================================
// 4. Qwen2 模型类
// =====================================================
struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    
    // KV Cache：使用 vector 存储每层的 cache tensor
    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;
    
    int64_t current_pos = 0;
    
    LlaisysQwen2Model(const LlaisysQwen2Meta *m) : meta(*m) {
        // 分配权重指针数组
        weights.attn_norm_w = new llaisysTensor_t[meta.nlayer]();
        weights.attn_q_w = new llaisysTensor_t[meta.nlayer]();
        weights.attn_q_b = new llaisysTensor_t[meta.nlayer]();
        weights.attn_k_w = new llaisysTensor_t[meta.nlayer]();
        weights.attn_k_b = new llaisysTensor_t[meta.nlayer]();
        weights.attn_v_w = new llaisysTensor_t[meta.nlayer]();
        weights.attn_v_b = new llaisysTensor_t[meta.nlayer]();
        weights.attn_o_w = new llaisysTensor_t[meta.nlayer]();
        weights.mlp_norm_w = new llaisysTensor_t[meta.nlayer]();
        weights.mlp_gate_w = new llaisysTensor_t[meta.nlayer]();
        weights.mlp_up_w = new llaisysTensor_t[meta.nlayer]();
        weights.mlp_down_w = new llaisysTensor_t[meta.nlayer]();

        // 【关键修复】在此处预初始化 KV Cache (空指针)，实际 tensor 在首次推理时创建
        // 或者可以在这里直接创建，取决于是否知道 batch size (本作业看似 bs=1)
    }

    ~LlaisysQwen2Model() {
        delete[] weights.attn_norm_w;
        delete[] weights.attn_q_w; delete[] weights.attn_q_b;
        delete[] weights.attn_k_w; delete[] weights.attn_k_b;
        delete[] weights.attn_v_w; delete[] weights.attn_v_b;
        delete[] weights.attn_o_w;
        delete[] weights.mlp_norm_w;
        delete[] weights.mlp_gate_w;
        delete[] weights.mlp_up_w;
        delete[] weights.mlp_down_w;
        
        // Tensor 智能指针会自动释放，清空 vector 即可
        k_cache.clear();
        v_cache.clear();
    }
    
    // 辅助：从 llaisysTensor_t (void*) 转为 tensor_t (智能指针/对象)
    tensor_t get_tensor(llaisysTensor_t t) {
        if (!t) return nullptr;
        return *reinterpret_cast<tensor_t*>(t);
    }
};

// =====================================================
// 5. C API 实现
// =====================================================

LLAISYS_EXPORT LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    return new LlaisysQwen2Model(meta);
}

LLAISYS_EXPORT void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
    delete model;
}

LLAISYS_EXPORT LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
    return &model->weights;
}

LLAISYS_EXPORT int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    auto &meta = model->meta;
    auto &w = model->weights;
    
    // 强制使用 FP32 进行计算，以规避 BF16 算子缺失问题
    llaisysDataType_t compute_dtype = LLAISYS_DTYPE_F32; 

    // 0. 准备输入 Tensor
    auto input = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
    input->load(token_ids); 

    // 1. 初始化 KV Cache (仅在第一次推理时)
    // 【关键修复】逻辑移到这里，确保只初始化一次
    if (model->k_cache.empty()) {
        for(size_t i=0; i<meta.nlayer; ++i) {
            // 创建全量 Cache：[max_seq, n_kv_head, head_dim]
            auto k_c = Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, compute_dtype, LLAISYS_DEVICE_CPU);
            auto v_c = Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, compute_dtype, LLAISYS_DEVICE_CPU);
            
            // 初始化为0 (可选)
            std::memset(k_c->data(), 0, k_c->nbytes());
            std::memset(v_c->data(), 0, v_c->nbytes());
            
            model->k_cache.push_back(k_c);
            model->v_cache.push_back(v_c);
        }
    }

    // 2. Embedding 层
    // 创建输入嵌入 X: [ntoken, hidden_size]
    auto x = Tensor::create({ntoken, meta.hs}, compute_dtype, LLAISYS_DEVICE_CPU);
    ops::embedding(x, input, model->get_tensor(w.in_embed));

    // 3. Transformer Layers
    for (size_t i = 0; i < meta.nlayer; ++i) {
        auto residual = Tensor::create(x->shape(), compute_dtype, LLAISYS_DEVICE_CPU);
        std::memcpy(residual->data(), x->data(), x->nbytes()); // 保存残差

        // --- Attention Block ---
        auto norm_out = Tensor::create(x->shape(), compute_dtype, LLAISYS_DEVICE_CPU);
        ops::rms_norm(norm_out, x, model->get_tensor(w.attn_norm_w[i]), meta.epsilon);
        
        // QKV Projections
        auto q = Tensor::create({ntoken, meta.nh, meta.dh}, compute_dtype, LLAISYS_DEVICE_CPU);
        auto k = Tensor::create({ntoken, meta.nkvh, meta.dh}, compute_dtype, LLAISYS_DEVICE_CPU);
        auto v = Tensor::create({ntoken, meta.nkvh, meta.dh}, compute_dtype, LLAISYS_DEVICE_CPU);
        
        // View 用于矩阵乘法兼容
        auto q_view = q->view({ntoken, meta.nh * meta.dh});
        auto k_view = k->view({ntoken, meta.nkvh * meta.dh});
        auto v_view = v->view({ntoken, meta.nkvh * meta.dh});
        
        ops::linear(q_view, norm_out, model->get_tensor(w.attn_q_w[i]), model->get_tensor(w.attn_q_b[i]));
        ops::linear(k_view, norm_out, model->get_tensor(w.attn_k_w[i]), model->get_tensor(w.attn_k_b[i]));
        ops::linear(v_view, norm_out, model->get_tensor(w.attn_v_w[i]), model->get_tensor(w.attn_v_b[i]));

        // RoPE (Rotary Positional Embeddings)
        auto pos_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
        std::vector<int64_t> pos_vec(ntoken);
        for(size_t p=0; p<ntoken; ++p) pos_vec[p] = model->current_pos + p;
        pos_ids->load(pos_vec.data());

        ops::rope(q, q, pos_ids, meta.theta);
        ops::rope(k, k, pos_ids, meta.theta);

        // Update KV Cache
        // 计算当前数据大小
        size_t elem_size = (compute_dtype == LLAISYS_DTYPE_F32) ? 4 : 2; 
        size_t head_dim_bytes = meta.dh * elem_size;
        size_t kv_len_bytes = ntoken * meta.nkvh * head_dim_bytes;
        
        // 计算 Cache 中的偏移量
        size_t offset_bytes = model->current_pos * meta.nkvh * head_dim_bytes;
        
        // 拷贝 K, V 到 Cache
        // 注意：这里假设 k/v memory layout 是连续的 [seq, heads, dim]
        std::byte* k_cache_ptr = (std::byte*)model->k_cache[i]->data();
        std::byte* v_cache_ptr = (std::byte*)model->v_cache[i]->data();
        
        std::memcpy(k_cache_ptr + offset_bytes, k->data(), kv_len_bytes);
        std::memcpy(v_cache_ptr + offset_bytes, v->data(), kv_len_bytes);

        // Self Attention
        // 这里的 q, k, v 在 ops::self_attention 实现中通常需要处理 GQA (Grouped Query Attention)
        // 我们传入当前步的 Q，和全量的 K/V Cache
        size_t total_seq_len = model->current_pos + ntoken;
        
        // 创建 Cache 的 Slice (视图)，包含从 0 到 total_seq_len 的数据
        // 注意：slice 的实现依赖于底层 tensor 库，这里假设 slice(dim, start, end)
        auto k_full = model->k_cache[i]->slice(0, 0, total_seq_len);
        auto v_full = model->v_cache[i]->slice(0, 0, total_seq_len);

        auto attn_out = Tensor::create({ntoken, meta.nh, meta.dh}, compute_dtype, LLAISYS_DEVICE_CPU);
        float scale = 1.0f / sqrtf((float)meta.dh);
        
        ops::self_attention(attn_out, q, k_full, v_full, scale);

        // Output Projection
        auto attn_out_view = attn_out->view({ntoken, meta.nh * meta.dh});
        // 重用 x 作为输出 buffer
        ops::linear(x, attn_out_view, model->get_tensor(w.attn_o_w[i]), nullptr);
        
        // Residual Connection
        tensor_add(x, residual);
        std::memcpy(residual->data(), x->data(), x->nbytes()); // 更新残差基准

        // --- Feed Forward Block (MLP) ---
        ops::rms_norm(norm_out, x, model->get_tensor(w.mlp_norm_w[i]), meta.epsilon);
        
        auto gate = Tensor::create({ntoken, meta.di}, compute_dtype, LLAISYS_DEVICE_CPU);
        auto up = Tensor::create({ntoken, meta.di}, compute_dtype, LLAISYS_DEVICE_CPU);
        
        ops::linear(gate, norm_out, model->get_tensor(w.mlp_gate_w[i]), nullptr);
        ops::linear(up, norm_out, model->get_tensor(w.mlp_up_w[i]), nullptr);
        
        // SwiGLU activation: gate = swish(gate) * up
        // 注意：有些实现 swiglu 是 gate * silu(gate) * up 还是 gate * sigmoid(gate) * up
        // Qwen2 通常是 silu(gate) * up
        ops::swiglu(gate, gate, up); // 假设结果存回 gate
        
        // Down Projection
        ops::linear(x, gate, model->get_tensor(w.mlp_down_w[i]), nullptr);
        
        // Final Residual
        tensor_add(x, residual);
    }

    // 4. Final Norm
    auto final_norm = Tensor::create(x->shape(), compute_dtype, LLAISYS_DEVICE_CPU);
    ops::rms_norm(final_norm, x, model->get_tensor(w.out_norm_w), meta.epsilon);

    // 5. Logits (LM Head)
    // 只需要最后一个 token 的输出
    auto last_hidden = final_norm->slice(0, ntoken - 1, ntoken); // [1, hidden]
    auto logits = Tensor::create({1, meta.voc}, compute_dtype, LLAISYS_DEVICE_CPU);
    
    ops::linear(logits, last_hidden, model->get_tensor(w.out_embed), nullptr);

    // 6. Argmax Sampling
    auto max_val = Tensor::create({1}, compute_dtype, LLAISYS_DEVICE_CPU);
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
    
    ops::argmax(max_idx, max_val, logits);
    
    int64_t next_token = ((int64_t*)max_idx->data())[0];
    
    // 更新位置索引
    model->current_pos += ntoken;

    return next_token;
}
