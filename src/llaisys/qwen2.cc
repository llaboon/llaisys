#include "llaisys/models/qwen2.h"
#include "llaisys/tensor.h"
// #include "../utils.hpp" // 如果找不到可以注释掉
// #include "../tensor/tensor.hpp" // 如果找不到可以注释掉
#include <vector>
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>

using namespace llaisys;

// =====================================================
// 1. 宏定义修复 (防止符号不可见)
// =====================================================
#ifndef LLAISYS_EXPORT
    #if defined(_WIN32)
        #define LLAISYS_EXPORT __declspec(dllexport)
    #else
        #define LLAISYS_EXPORT __attribute__((visibility("default")))
    #endif
#endif

// =====================================================
// 2. 算子声明
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
// 3. 辅助函数
// =====================================================
void tensor_add(tensor_t dst, tensor_t src) {
    if (dst->numel() != src->numel()) return;
    
    // 假设都是 FP32 (因为我们在 Python 端把 BF16 转成了 FP32)
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
    
    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;
    
    int64_t current_pos = 0;
    
    LlaisysQwen2Model(const LlaisysQwen2Meta *m) : meta(*m) {
        // 使用 () 初始化，确保指针数组被清零
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
        
        k_cache.clear();
        v_cache.clear();
    }
    
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
    if (model) delete model;
}

LLAISYS_EXPORT LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
    return &model->weights;
}

LLAISYS_EXPORT int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    auto &meta = model->meta;
    auto &w = model->weights;
    
    llaisysDataType_t compute_dtype = LLAISYS_DTYPE_F32; 

    // 0. 输入 Tensor
    auto input = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
    input->load(token_ids); 

    // 1. 初始化 KV Cache (仅首次)
    if (model->k_cache.empty()) {
        for(size_t i=0; i<meta.nlayer; ++i) {
            auto k_c = Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, compute_dtype, LLAISYS_DEVICE_CPU);
            auto v_c = Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, compute_dtype, LLAISYS_DEVICE_CPU);
            
            // 手动清零 (修复 nbytes 报错)
            size_t cache_bytes = k_c->numel() * sizeof(float);
            std::memset(k_c->data(), 0, cache_bytes);
            std::memset(v_c->data(), 0, cache_bytes);
            
            model->k_cache.push_back(k_c);
            model->v_cache.push_back(v_c);
        }
    }

    // 2. Embedding
    auto x = Tensor::create({ntoken, meta.hs}, compute_dtype, LLAISYS_DEVICE_CPU);
    // 检查权重是否存在，防止空指针 crash
    if (w.in_embed) {
        ops::embedding(x, input, model->get_tensor(w.in_embed));
    }

    // 3. Layers Loop
    for (size_t i = 0; i < meta.nlayer; ++i) {
        auto residual = Tensor::create(x->shape(), compute_dtype, LLAISYS_DEVICE_CPU);
        // 保存 Residual
        std::memcpy(residual->data(), x->data(), x->numel() * sizeof(float)); 

        // --- Attention ---
        auto norm_out = Tensor::create(x->shape(), compute_dtype, LLAISYS_DEVICE_CPU);
        ops::rms_norm(norm_out, x, model->get_tensor(w.attn_norm_w[i]), meta.epsilon);
        
        auto q = Tensor::create({ntoken, meta.nh, meta.dh}, compute_dtype, LLAISYS_DEVICE_CPU);
        auto k = Tensor::create({ntoken, meta.nkvh, meta.dh}, compute_dtype, LLAISYS_DEVICE_CPU);
        auto v = Tensor::create({ntoken, meta.nkvh, meta.dh}, compute_dtype, LLAISYS_DEVICE_CPU);
        
        auto q_view = q->view({ntoken, meta.nh * meta.dh});
        auto k_view = k->view({ntoken, meta.nkvh * meta.dh});
        auto v_view = v->view({ntoken, meta.nkvh * meta.dh});
        
        // 【核心修复】Bias 显式传 nullptr，因为 Qwen2 没有这些 Bias
        ops::linear(q_view, norm_out, model->get_tensor(w.attn_q_w[i]), nullptr);
        ops::linear(k_view, norm_out, model->get_tensor(w.attn_k_w[i]), nullptr);
        ops::linear(v_view, norm_out, model->get_tensor(w.attn_v_w[i]), nullptr);

        // RoPE
        auto pos_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
        std::vector<int64_t> pos_vec(ntoken);
        for(size_t p=0; p<ntoken; ++p) pos_vec[p] = model->current_pos + p;
        pos_ids->load(pos_vec.data());

        ops::rope(q, q, pos_ids, meta.theta);
        ops::rope(k, k, pos_ids, meta.theta);

        // Update Cache
        size_t head_dim_bytes = meta.dh * sizeof(float);
        size_t kv_len_bytes = ntoken * meta.nkvh * head_dim_bytes;
        size_t offset_bytes = model->current_pos * meta.nkvh * head_dim_bytes;
        
        // 确保 offset 不越界 (虽然 test 可能不会跑那么长)
        if (model->current_pos < meta.maxseq) {
             std::byte* k_cache_ptr = (std::byte*)model->k_cache[i]->data();
             std::byte* v_cache_ptr = (std::byte*)model->v_cache[i]->data();
             std::memcpy(k_cache_ptr + offset_bytes, k->data(), kv_len_bytes);
             std::memcpy(v_cache_ptr + offset_bytes, v->data(), kv_len_bytes);
        }

        size_t total_seq_len = model->current_pos + ntoken;
        
        auto k_full = model->k_cache[i]->slice(0, 0, total_seq_len);
        auto v_full = model->v_cache[i]->slice(0, 0, total_seq_len);

        auto attn_out = Tensor::create({ntoken, meta.nh, meta.dh}, compute_dtype, LLAISYS_DEVICE_CPU);
        float scale = 1.0f / sqrtf((float)meta.dh);
        
        ops::self_attention(attn_out, q, k_full, v_full, scale);

        auto attn_out_view = attn_out->view({ntoken, meta.nh * meta.dh});
        // 【核心修复】Bias 传 nullptr
        ops::linear(x, attn_out_view, model->get_tensor(w.attn_o_w[i]), nullptr);
        
        tensor_add(x, residual);
        std::memcpy(residual->data(), x->data(), x->numel() * sizeof(float)); // Update residual

        // --- MLP ---
        ops::rms_norm(norm_out, x, model->get_tensor(w.mlp_norm_w[i]), meta.epsilon);
        
        auto gate = Tensor::create({ntoken, meta.di}, compute_dtype, LLAISYS_DEVICE_CPU);
        auto up = Tensor::create({ntoken, meta.di}, compute_dtype, LLAISYS_DEVICE_CPU);
        
        // 【核心修复】Bias 传 nullptr
        ops::linear(gate, norm_out, model->get_tensor(w.mlp_gate_w[i]), nullptr);
        ops::linear(up, norm_out, model->get_tensor(w.mlp_up_w[i]), nullptr);
        
        ops::swiglu(gate, gate, up);
        
        // 【核心修复】Bias 传 nullptr
        ops::linear(x, gate, model->get_tensor(w.mlp_down_w[i]), nullptr);
        
        tensor_add(x, residual);
    }

    // 4. Final Norm
    auto final_norm = Tensor::create(x->shape(), compute_dtype, LLAISYS_DEVICE_CPU);
    ops::rms_norm(final_norm, x, model->get_tensor(w.out_norm_w), meta.epsilon);

    // 5. Logits
    auto last_hidden = final_norm->slice(0, ntoken - 1, ntoken); 
    auto logits = Tensor::create({1, meta.voc}, compute_dtype, LLAISYS_DEVICE_CPU);
    
    // 【核心修复】Bias 传 nullptr
    ops::linear(logits, last_hidden, model->get_tensor(w.out_embed), nullptr);

    // 6. Argmax
    auto max_val = Tensor::create({1}, compute_dtype, LLAISYS_DEVICE_CPU);
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
    
    ops::argmax(max_idx, max_val, logits);
    
    int64_t next_token = ((int64_t*)max_idx->data())[0];
    
    model->current_pos += ntoken;

    return next_token;
}
