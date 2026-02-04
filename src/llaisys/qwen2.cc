#include "llaisys/models/qwen2.h"
#include "llaisys/tensor.h"

// 必须包含 Tensor 类定义，因为 tensor_t 是 shared_ptr<Tensor>
#include "../tensor/tensor.hpp"

#include <vector>
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <memory> 
#include <cstddef> // for std::byte

using namespace llaisys;

// 宏定义
#ifndef LLAISYS_EXPORT
    #if defined(_WIN32)
        #define LLAISYS_EXPORT __declspec(dllexport)
    #else
        #define LLAISYS_EXPORT __attribute__((visibility("default")))
    #endif
#endif

// =====================================================
// 算子声明
// =====================================================
namespace llaisys::ops {
    // tensor_t 已经在 tensor.hpp 中定义为 std::shared_ptr<Tensor>
    void embedding(tensor_t out, tensor_t index, tensor_t weight);
    void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps);
    void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);
    void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);
    void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
    void swiglu(tensor_t out, tensor_t gate, tensor_t up);
    void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
}

// =====================================================
// 辅助函数
// =====================================================
void tensor_add(tensor_t dst, tensor_t src) {
    if (dst->numel() != src->numel()) return;
    
    if (dst->dtype() == LLAISYS_DTYPE_F32) {
        float* d = (float*)dst->data();
        float* s = (float*)src->data();
        for(size_t i=0; i<dst->numel(); ++i) d[i] += s[i];
    }
}

// =====================================================
// Qwen2 模型类
// =====================================================
struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    
    // KV Cache 使用智能指针，无需手动管理内存
    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;
    
    int64_t current_pos = 0;
    
    LlaisysQwen2Model(const LlaisysQwen2Meta *m) : meta(*m) {
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
        
        // k_cache, v_cache 是 vector<shared_ptr>，会自动释放，不需要手动 delete
        k_cache.clear();
        v_cache.clear();
    }
    
    // 【核心修复】将原始指针封装为 shared_ptr，但使用空删除器
    // 这样既满足了 tensor_t (shared_ptr) 的类型要求，又不会 double free
    tensor_t get_tensor(llaisysTensor_t t) {
        if (!t) return nullptr;
        
        // t 是 Tensor* (raw pointer)
        // 创建一个指向它的 shared_ptr，但 deleter 里面什么都不做
        return std::shared_ptr<Tensor>(reinterpret_cast<Tensor*>(t), [](Tensor*){});
    }
};

// =====================================================
// C API 实现
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
    // Tensor::create 返回的就是 tensor_t (shared_ptr)，会自动管理内存
    auto input = Tensor::create({(size_t)ntoken}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
    input->load(token_ids); 

    // 1. 初始化 KV Cache
    if (model->k_cache.empty()) {
        for(size_t i=0; i<meta.nlayer; ++i) {
            auto k_c = Tensor::create({(size_t)meta.maxseq, (size_t)meta.nkvh, (size_t)meta.dh}, compute_dtype, LLAISYS_DEVICE_CPU);
            auto v_c = Tensor::create({(size_t)meta.maxseq, (size_t)meta.nkvh, (size_t)meta.dh}, compute_dtype, LLAISYS_DEVICE_CPU);
            
            size_t cache_bytes = k_c->numel() * sizeof(float);
            std::memset(k_c->data(), 0, cache_bytes);
            std::memset(v_c->data(), 0, cache_bytes);
            
            model->k_cache.push_back(k_c);
            model->v_cache.push_back(v_c);
        }
    }

    // 2. Embedding
    auto x = Tensor::create({(size_t)ntoken, (size_t)meta.hs}, compute_dtype, LLAISYS_DEVICE_CPU);
    if (w.in_embed) {
        ops::embedding(x, input, model->get_tensor(w.in_embed));
    }

    // 3. Layers Loop
    for (size_t i = 0; i < meta.nlayer; ++i) {
        auto residual = Tensor::create(x->shape(), compute_dtype, LLAISYS_DEVICE_CPU);
        std::memcpy(residual->data(), x->data(), x->numel() * sizeof(float)); 

        // --- Attention ---
        auto norm_out = Tensor::create(x->shape(), compute_dtype, LLAISYS_DEVICE_CPU);
        ops::rms_norm(norm_out, x, model->get_tensor(w.attn_norm_w[i]), meta.epsilon);
        
        auto q = Tensor::create({(size_t)ntoken, (size_t)meta.nh, (size_t)meta.dh}, compute_dtype, LLAISYS_DEVICE_CPU);
        auto k = Tensor::create({(size_t)ntoken, (size_t)meta.nkvh, (size_t)meta.dh}, compute_dtype, LLAISYS_DEVICE_CPU);
        auto v = Tensor::create({(size_t)ntoken, (size_t)meta.nkvh, (size_t)meta.dh}, compute_dtype, LLAISYS_DEVICE_CPU);
        
        auto q_view = q->view({(size_t)ntoken, (size_t)(meta.nh * meta.dh)});
        auto k_view = k->view({(size_t)ntoken, (size_t)(meta.nkvh * meta.dh)});
        auto v_view = v->view({(size_t)ntoken, (size_t)(meta.nkvh * meta.dh)});
        
        ops::linear(q_view, norm_out, model->get_tensor(w.attn_q_w[i]), nullptr);
        ops::linear(k_view, norm_out, model->get_tensor(w.attn_k_w[i]), nullptr);
        ops::linear(v_view, norm_out, model->get_tensor(w.attn_v_w[i]), nullptr);

        // RoPE
        auto pos_ids = Tensor::create({(size_t)ntoken}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
        std::vector<int64_t> pos_vec(ntoken);
        for(size_t p=0; p<ntoken; ++p) pos_vec[p] = model->current_pos + p;
        pos_ids->load(pos_vec.data());

        ops::rope(q, q, pos_ids, meta.theta);
        ops::rope(k, k, pos_ids, meta.theta);

        // Update Cache
        size_t head_dim_bytes = meta.dh * sizeof(float);
        size_t kv_len_bytes = ntoken * meta.nkvh * head_dim_bytes;
        size_t offset_bytes = model->current_pos * meta.nkvh * head_dim_bytes;
        
        if ((size_t)model->current_pos < meta.maxseq) {
             std::byte* k_cache_ptr = (std::byte*)model->k_cache[i]->data();
             std::byte* v_cache_ptr = (std::byte*)model->v_cache[i]->data();
             std::memcpy(k_cache_ptr + offset_bytes, k->data(), kv_len_bytes);
             std::memcpy(v_cache_ptr + offset_bytes, v->data(), kv_len_bytes);
        }

        size_t total_seq_len = model->current_pos + ntoken;
        
        auto k_full = model->k_cache[i]->slice(0, 0, (size_t)total_seq_len);
        auto v_full = model->v_cache[i]->slice(0, 0, (size_t)total_seq_len);

        auto attn_out = Tensor::create({(size_t)ntoken, (size_t)meta.nh, (size_t)meta.dh}, compute_dtype, LLAISYS_DEVICE_CPU);
        float scale = 1.0f / sqrtf((float)meta.dh);
        
        ops::self_attention(attn_out, q, k_full, v_full, scale);

        auto attn_out_view = attn_out->view({(size_t)ntoken, (size_t)(meta.nh * meta.dh)});
        ops::linear(x, attn_out_view, model->get_tensor(w.attn_o_w[i]), nullptr);
        
        tensor_add(x, residual);
        std::memcpy(residual->data(), x->data(), x->numel() * sizeof(float)); 

        // --- MLP ---
        ops::rms_norm(norm_out, x, model->get_tensor(w.mlp_norm_w[i]), meta.epsilon);
        
        auto gate = Tensor::create({(size_t)ntoken, (size_t)meta.di}, compute_dtype, LLAISYS_DEVICE_CPU);
        auto up = Tensor::create({(size_t)ntoken, (size_t)meta.di}, compute_dtype, LLAISYS_DEVICE_CPU);
        
        ops::linear(gate, norm_out, model->get_tensor(w.mlp_gate_w[i]), nullptr);
        ops::linear(up, norm_out, model->get_tensor(w.mlp_up_w[i]), nullptr);
        
        ops::swiglu(gate, gate, up);
        
        ops::linear(x, gate, model->get_tensor(w.mlp_down_w[i]), nullptr);
        
        tensor_add(x, residual);
    }

    // 4. Final Norm
    auto final_norm = Tensor::create(x->shape(), compute_dtype, LLAISYS_DEVICE_CPU);
    ops::rms_norm(final_norm, x, model->get_tensor(w.out_norm_w), meta.epsilon);

    // 5. Logits
    auto last_hidden = final_norm->slice(0, (size_t)ntoken - 1, (size_t)ntoken); 
    auto logits = Tensor::create({1, (size_t)meta.voc}, compute_dtype, LLAISYS_DEVICE_CPU);
    
    ops::linear(logits, last_hidden, model->get_tensor(w.out_embed), nullptr);

    // 6. Argmax
    auto max_val = Tensor::create({1}, compute_dtype, LLAISYS_DEVICE_CPU);
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
    
    ops::argmax(max_idx, max_val, logits);
    
    int64_t next_token = ((int64_t*)max_idx->data())[0];
    
    model->current_pos += ntoken;

    return next_token;
}
