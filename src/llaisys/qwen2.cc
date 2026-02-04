#include "llaisys/models/qwen2.h"
#include "llaisys/tensor.h"
#include "../tensor/tensor.hpp"

#include <vector>
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <memory>

using namespace llaisys;

// =====================================================
// 宏定义与辅助
// =====================================================
#ifndef LLAISYS_EXPORT
    #if defined(_WIN32)
        #define LLAISYS_EXPORT __declspec(dllexport)
    #else
        #define LLAISYS_EXPORT __attribute__((visibility("default")))
    #endif
#endif

// =====================================================
// 算子声明 (与 Python 绑定一致)
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

// 简单的 Tensor 加法辅助
void tensor_add(tensor_t dst, tensor_t src) {
    if (dst->numel() != src->numel()) return;
    if (dst->dtype() == LLAISYS_DTYPE_F32) {
        float* d = (float*)dst->data();
        float* s = (float*)src->data();
        for(size_t i=0; i<dst->numel(); ++i) d[i] += s[i];
    }
}

// =====================================================
// 关键修复：类型转换辅助函数
// =====================================================
// 将 void* (实际上是 Tensor**) 转换为 C++ 可用的 shared_ptr<Tensor>
tensor_t get_tensor(void* t) {
    if (!t) return nullptr;
    // 双重指针解引用： void* -> Tensor** -> Tensor*
    Tensor* real_ptr = *(Tensor**)t;
    // 包装成 shared_ptr，不接管所有权 (Empty Deleter)
    return std::shared_ptr<Tensor>(real_ptr, [](Tensor*){});
}

// =====================================================
// 模型结构体
// =====================================================
struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;

    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;
    int64_t current_pos = 0;

    LlaisysQwen2Model(const LlaisysQwen2Meta *m) : meta(*m) {
        // 分配指针数组
        weights.attn_norm_w = new void*[meta.nlayer]();
        weights.attn_q_w    = new void*[meta.nlayer]();
        weights.attn_q_b    = new void*[meta.nlayer]();
        weights.attn_k_w    = new void*[meta.nlayer]();
        weights.attn_k_b    = new void*[meta.nlayer]();
        weights.attn_v_w    = new void*[meta.nlayer]();
        weights.attn_v_b    = new void*[meta.nlayer]();
        weights.attn_o_w    = new void*[meta.nlayer]();
        weights.mlp_norm_w  = new void*[meta.nlayer]();
        weights.mlp_gate_w  = new void*[meta.nlayer]();
        weights.mlp_up_w    = new void*[meta.nlayer]();
        weights.mlp_down_w  = new void*[meta.nlayer]();
    }

    ~LlaisysQwen2Model() {
        // 释放指针数组
        delete[] (void**)weights.attn_norm_w;
        delete[] (void**)weights.attn_q_w; delete[] (void**)weights.attn_q_b;
        delete[] (void**)weights.attn_k_w; delete[] (void**)weights.attn_k_b;
        delete[] (void**)weights.attn_v_w; delete[] (void**)weights.attn_v_b;
        delete[] (void**)weights.attn_o_w;
        delete[] (void**)weights.mlp_norm_w;
        delete[] (void**)weights.mlp_gate_w;
        delete[] (void**)weights.mlp_up_w;
        delete[] (void**)weights.mlp_down_w;
        k_cache.clear();
        v_cache.clear();
    }
};

// =====================================================
// 导出 API
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

// =====================================================
// 推理核心函数 (包含 Debug 逻辑)
// =====================================================
LLAISYS_EXPORT int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    std::cerr << ">>> [Debug] 1. Function Start. ntoken=" << ntoken << std::endl;

    if (!model) return 0;
    if (ntoken == 0) return model->meta.end_token;

    auto &meta = model->meta;
    auto &w = model->weights;
    llaisysDataType_t compute_dtype = LLAISYS_DTYPE_F32;

    // 1. Input
    std::cerr << ">>> [Debug] 2. Creating Input Tensor..." << std::endl;
    auto input = Tensor::create({(size_t)ntoken}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
    if (!token_ids) { std::cerr << "FATAL: token_ids null" << std::endl; return 0; }
    input->load(token_ids);

    // 2. KV Cache
    if (model->k_cache.empty()) {
        std::cerr << ">>> [Debug] 5. Init KV Cache..." << std::endl;
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

    // 3. Embedding
    std::cerr << ">>> [Debug] 6. Embedding..." << std::endl;
    auto x = Tensor::create({(size_t)ntoken, (size_t)meta.hs}, compute_dtype, LLAISYS_DEVICE_CPU);

    if (!w.in_embed) { std::cerr << "FATAL: in_embed is NULL" << std::endl; return 0; }
    // 使用 get_tensor 处理双重指针
    ops::embedding(x, input, get_tensor(w.in_embed));
    std::cerr << ">>> [Debug] 9. Embedding done." << std::endl;

    // 4. Layers Loop
    for (size_t i = 0; i < meta.nlayer; ++i) {
        // std::cerr << ">>> [Debug] Loop Layer " << i << " Start" << std::endl; // 减少刷屏，注释掉

        auto residual = Tensor::create(x->shape(), compute_dtype, LLAISYS_DEVICE_CPU);
        std::memcpy(residual->data(), x->data(), x->numel() * sizeof(float));

        // Attention Norm
        if (w.attn_norm_w[i] == nullptr) { std::cerr << "FATAL: L" << i << " attn_norm_w is NULL" << std::endl; return 0; }
        auto norm_out = Tensor::create(x->shape(), compute_dtype, LLAISYS_DEVICE_CPU);
        ops::rms_norm(norm_out, x, get_tensor(w.attn_norm_w[i]), meta.epsilon);

        // QKV
        auto q = Tensor::create({(size_t)ntoken, (size_t)meta.nh, (size_t)meta.dh}, compute_dtype, LLAISYS_DEVICE_CPU);
        auto k = Tensor::create({(size_t)ntoken, (size_t)meta.nkvh, (size_t)meta.dh}, compute_dtype, LLAISYS_DEVICE_CPU);
        auto v = Tensor::create({(size_t)ntoken, (size_t)meta.nkvh, (size_t)meta.dh}, compute_dtype, LLAISYS_DEVICE_CPU);

        auto q_view = q->view({(size_t)ntoken, (size_t)(meta.nh * meta.dh)});
        auto k_view = k->view({(size_t)ntoken, (size_t)(meta.nkvh * meta.dh)});
        auto v_view = v->view({(size_t)ntoken, (size_t)(meta.nkvh * meta.dh)});

        ops::linear(q_view, norm_out, get_tensor(w.attn_q_w[i]), nullptr);
        ops::linear(k_view, norm_out, get_tensor(w.attn_k_w[i]), nullptr);
        ops::linear(v_view, norm_out, get_tensor(w.attn_v_w[i]), nullptr);

        // RoPE
        auto pos_ids = Tensor::create({(size_t)ntoken}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
        std::vector<int64_t> pos_vec(ntoken);
        for(size_t p=0; p<ntoken; ++p) pos_vec[p] = model->current_pos + p;
        pos_ids->load(pos_vec.data());
        ops::rope(q, q, pos_ids, meta.theta);
        ops::rope(k, k, pos_ids, meta.theta);

        // Cache Update
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

        // Attention
        auto attn_out = Tensor::create({(size_t)ntoken, (size_t)meta.nh, (size_t)meta.dh}, compute_dtype, LLAISYS_DEVICE_CPU);
        float scale = 1.0f / sqrtf((float)meta.dh);
        ops::self_attention(attn_out, q, k_full, v_full, scale);

        // Output Linear
        auto attn_out_view = attn_out->view({(size_t)ntoken, (size_t)(meta.nh * meta.dh)});
        ops::linear(x, attn_out_view, get_tensor(w.attn_o_w[i]), nullptr);

        tensor_add(x, residual);
        std::memcpy(residual->data(), x->data(), x->numel() * sizeof(float));

        // MLP
        ops::rms_norm(norm_out, x, get_tensor(w.mlp_norm_w[i]), meta.epsilon);

        auto gate = Tensor::create({(size_t)ntoken, (size_t)meta.di}, compute_dtype, LLAISYS_DEVICE_CPU);
        auto up = Tensor::create({(size_t)ntoken, (size_t)meta.di}, compute_dtype, LLAISYS_DEVICE_CPU);

        ops::linear(gate, norm_out, get_tensor(w.mlp_gate_w[i]), nullptr);
        ops::linear(up, norm_out, get_tensor(w.mlp_up_w[i]), nullptr);

        ops::swiglu(gate, gate, up);

        ops::linear(x, gate, get_tensor(w.mlp_down_w[i]), nullptr);
        tensor_add(x, residual);
    }

    std::cerr << ">>> [Debug] 11. Loop Finished. Entering Final Stage..." << std::endl;

    // 5. Final Norm
    std::cerr << ">>> [Debug] 12. Final Norm..." << std::endl;
    if (!w.out_norm_w) { std::cerr << "FATAL: out_norm_w NULL" << std::endl; return 0; }

    // 【定义 final_norm】
    auto final_norm = Tensor::create(x->shape(), compute_dtype, LLAISYS_DEVICE_CPU);
    ops::rms_norm(final_norm, x, get_tensor(w.out_norm_w), meta.epsilon);
    std::cerr << ">>> [Debug] 12. Final Norm Done." << std::endl;

    // 6. Logits (LM Head) - 包含探测代码
    std::cerr << ">>> [Debug] 13. Computing Logits..." << std::endl;

    if (!w.out_embed) { std::cerr << "FATAL: out_embed is NULL" << std::endl; return 0; }
    std::cerr << ">>> [Debug] Checking out_embed ptr: " << w.out_embed << std::endl;

    // 核酸检测
    Tensor* lm_head_ptr = *(Tensor**)w.out_embed;
    std::cerr << ">>> [Debug] PROBE Dereferenced: " << lm_head_ptr << std::endl;
    // 如果这里崩了，说明 out_embed 指针坏了
    std::cerr << ">>> [Debug] PROBE ndim: " << lm_head_ptr->shape().size() << std::endl;

// ... (Final Norm Done 之后) ...
    // ... (Final Norm Done 之前代码保持不变) ...
    std::cerr << ">>> [Debug] 12. Final Norm Done." << std::endl;

    // --- 2. Logits (LM Head) ---
    std::cerr << ">>> [Debug] 13. Computing Logits (Manual Implementation)..." << std::endl;

    // 1. 准备 Input (Last Hidden State)
    auto last_hidden = Tensor::create({1, (size_t)meta.hs}, compute_dtype, LLAISYS_DEVICE_CPU);

    // [检查] 确保 last_hidden 分配成功
    if (last_hidden->data() == nullptr) {
        std::cerr << ">>> [FATAL] last_hidden allocation failed (data is null)!" << std::endl;
        return 0;
    }

    size_t last_token_offset = (ntoken - 1) * meta.hs;
    float* src_ptr = (float*)final_norm->data() + last_token_offset;
    float* hidden_ptr = (float*)last_hidden->data();

    // [检查] 确保 src_ptr 有效 (final_norm 是否有数据)
    if (final_norm->data() == nullptr) {
         std::cerr << ">>> [FATAL] final_norm data is null!" << std::endl;
         return 0;
    }

    std::memcpy(hidden_ptr, src_ptr, meta.hs * sizeof(float));

    // 2. 准备 Weight (Out Embed)
    Tensor* weight_tensor = *(Tensor**)w.out_embed;

    // [关键检查] 检查权重数据指针！
    void* raw_weight_data = weight_tensor->data();
    std::cerr << ">>> [Debug] Weight Tensor Ptr: " << weight_tensor << std::endl;
    std::cerr << ">>> [Debug] Weight Data Ptr: " << raw_weight_data << std::endl;

    if (raw_weight_data == nullptr) {
        std::cerr << ">>> [FATAL] Weight data is NULL! Python loaded shape but NO DATA." << std::endl;
        // 这是一个常见问题：显存/内存不足，或者 Python load 没调用成功
        return 0;
    }
    float* weight_data = (float*)raw_weight_data;

    // 3. 准备 Output (Logits)
    auto logits = Tensor::create({1, (size_t)meta.voc}, compute_dtype, LLAISYS_DEVICE_CPU);
    float* logits_data = (float*)logits->data();

    if (logits_data == nullptr) {
        std::cerr << ">>> [FATAL] Logits output allocation failed!" << std::endl;
        return 0;
    }

    // 4. 维度检查
    size_t vocab_size = meta.voc;
    size_t hidden_size = meta.hs;
    size_t weight_dim0 = weight_tensor->shape()[0];
    size_t weight_dim1 = weight_tensor->shape()[1];

    // 修正维度 (以权重实际大小为准)
    if (weight_dim0 != vocab_size) vocab_size = weight_dim0;
    if (weight_dim1 != hidden_size) hidden_size = weight_dim1;

    std::cerr << ">>> [Debug] Running MatMul Loop (Vocab=" << vocab_size << ", Hidden=" << hidden_size << ")..." << std::endl;

    // 5. 手动执行矩阵乘法
    for (size_t v = 0; v < vocab_size; ++v) {
        float sum = 0.0f;
        float* current_weight_row = weight_data + (v * hidden_size);

        for (size_t h = 0; h < hidden_size; ++h) {
            sum += hidden_ptr[h] * current_weight_row[h];
        }
        logits_data[v] = sum;

        // [可选] 打印第一个结果证明活过来了
        if (v == 0) std::cerr << ">>> [Debug] First Logit Calculated: " << sum << std::endl;
    }
    std::cerr << ">>> [Debug] 13. Logits Done (Manually)." << std::endl;

    // --- 3. Argmax ---
    std::cerr << ">>> [Debug] 14. Argmax..." << std::endl;
    auto max_val = Tensor::create({1}, compute_dtype, LLAISYS_DEVICE_CPU);
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);

    ops::argmax(max_idx, max_val, logits);

    int64_t next_token = ((int64_t*)max_idx->data())[0];

    std::cerr << ">>> [Debug] 15. Success! Next Token: " << next_token << std::endl;

    model->current_pos += ntoken;

    return next_token;
}
