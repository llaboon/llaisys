#include "llaisys/models/qwen2.h"
#include "llaisys/tensor.h"
#include "../ops/op.hpp" // 引用之前的算子
#include <vector>
#include <iostream>
#include <cstring>
#include <cmath>

using namespace llaisys;

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    
    // KV Cache: [layer][k/v]
    // 为了简单，我们直接存 Tensor 对象
    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;
    
    int64_t current_pos = 0;
    
    // 构造函数：分配权重数组空间和 KV Cache
    LlaisysQwen2Model(const LlaisysQwen2Meta *m) : meta(*m) {
        // 1. 分配权重指针数组
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

        // 2. 初始化 KV Cache (这里仅仅是占位，真正分配需要在 infer 中拿到 dtype 后，或者这里默认 fp32/bf16)
        // 假设我们会在第一次 infer 时根据权重 dtype 初始化 cache
    }

    ~LlaisysQwen2Model() {
        delete[] weights.attn_norm_w;
        delete[] weights.attn_q_w;
        delete[] weights.attn_q_b;
        delete[] weights.attn_k_w;
        delete[] weights.attn_k_b;
        delete[] weights.attn_v_w;
        delete[] weights.attn_v_b;
        delete[] weights.attn_o_w;
        delete[] weights.mlp_norm_w;
        delete[] weights.mlp_gate_w;
        delete[] weights.mlp_up_w;
        delete[] weights.mlp_down_w;
    }
    
    // 辅助：获取 Tensor 指针 (llaisysTensor_t 是 void*，强转为 tensor_t 智能指针)
    tensor_t get_tensor(llaisysTensor_t t) {
        if (!t) return nullptr;
        // 假设 llaisysTensor_t 是 tensor_t (std::shared_ptr<Tensor>) 的 raw pointer 或者直接就是对象
        // 根据 tensor.h 的定义，通常它是 void* 指向 std::shared_ptr<Tensor>
        return *(tensor_t*)t;
    }
};

// C API 实现
extern "C" {

LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    return new LlaisysQwen2Model(meta);
}

void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
    delete model;
}

LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
    return &model->weights;
}

int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    auto &meta = model->meta;
    auto &w = model->weights;
    
    // 0. 准备 Input Tensor
    // 必须拷贝 token_ids 到 Tensor
    auto input = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
    input->load(token_ids); // Copy from host

    // 获取 embedding table 用于确定 dtype
    auto embed_w = model->get_tensor(w.in_embed);
    auto dtype = embed_w->dtype();
    
    // 初始化 KV Cache (如果还没初始化)
    if (model->k_cache.empty()) {
        for(size_t i=0; i<meta.nlayer; ++i) {
            model->k_cache.push_back(Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, dtype, LLAISYS_DEVICE_CPU));
            model->v_cache.push_back(Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, dtype, LLAISYS_DEVICE_CPU));
        }
    }

    // 1. Embedding
    auto x = Tensor::create({ntoken, meta.hs}, dtype, LLAISYS_DEVICE_CPU);
    ops::embedding(x, input, embed_w);

    // 2. Layers Loop
    for (size_t i = 0; i < meta.nlayer; ++i) {
        auto residual = x; // 记录残差
        
        // --- Attention Block ---
        // RMS Norm
        auto norm_out = Tensor::create(x->shape(), dtype, LLAISYS_DEVICE_CPU);
        ops::rms_norm(norm_out, x, model->get_tensor(w.attn_norm_w[i]), meta.epsilon);
        
        // QKV Proj
        // 注意：Linear 算子会自动处理 shape
        // Q: [ntoken, nh, dh]
        auto q = Tensor::create({ntoken, meta.nh, meta.dh}, dtype, LLAISYS_DEVICE_CPU);
        auto k = Tensor::create({ntoken, meta.nkvh, meta.dh}, dtype, LLAISYS_DEVICE_CPU);
        auto v = Tensor::create({ntoken, meta.nkvh, meta.dh}, dtype, LLAISYS_DEVICE_CPU);
        
        // Flatten for Linear: [ntoken, dim]
        auto q_view = q->view({ntoken, meta.nh * meta.dh});
        auto k_view = k->view({ntoken, meta.nkvh * meta.dh});
        auto v_view = v->view({ntoken, meta.nkvh * meta.dh});
        
        ops::linear(q_view, norm_out, model->get_tensor(w.attn_q_w[i]), model->get_tensor(w.attn_q_b[i]));
        ops::linear(k_view, norm_out, model->get_tensor(w.attn_k_w[i]), model->get_tensor(w.attn_k_b[i]));
        ops::linear(v_view, norm_out, model->get_tensor(w.attn_v_w[i]), model->get_tensor(w.attn_v_b[i]));
        
        // RoPE
        // 构造 pos_ids [start, ..., start + ntoken]
        auto pos_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
        std::vector<int64_t> pos_vec(ntoken);
        for(size_t p=0; p<ntoken; ++p) pos_vec[p] = model->current_pos + p;
        pos_ids->load(pos_vec.data());
        
        ops::rope(q, q, pos_ids, meta.theta);
        ops::rope(k, k, pos_ids, meta.theta);
        
        // Update KV Cache
        // 将当前 k, v 拷贝到 cache 的 [current_pos] 位置
        // 简单内存拷贝 (需要根据 dtype 计算字节大小)
        size_t element_size = q->elementSize();
        size_t k_bytes = ntoken * meta.nkvh * meta.dh * element_size;
        
        // 获取 Cache 中对应位置的指针
        uint8_t* k_cache_ptr = (uint8_t*)model->k_cache[i]->data() + model->current_pos * meta.nkvh * meta.dh * element_size;
        uint8_t* v_cache_ptr = (uint8_t*)model->v_cache[i]->data() + model->current_pos * meta.nkvh * meta.dh * element_size;
        
        memcpy(k_cache_ptr, k->data(), k_bytes);
        memcpy(v_cache_ptr, v->data(), k_bytes); // v size same as k
        
        // Prepare K, V for Attention (View of Cache up to current_pos + ntoken)
        // 这里的 Slice 必须是 contiguous 的，或者 self_attention 支持 stride
        // 假设 self_attention 接受的 k, v 是 [total_len, nkvh, dh]
        // 我们利用 slice (假设作业2实现了 slice) 或者直接创建一个 view 指向 cache 开头
        // 为了简单，我们只传递 cache 本身，但在 self_attention 内部，total_len 参数控制读取长度
        size_t total_len = model->current_pos + ntoken;
        
        auto k_total = model->k_cache[i]; // 全量 cache
        auto v_total = model->v_cache[i];
        // *关键*：需要告诉 self_attention 真实的有效长度是 total_len，而不是 max_seq
        // 作业2的 self_attention 原型是 (attn_val, q, k, v, scale)
        // 它内部使用 k->shape[0] 作为 total_len。所以我们必须 slice 出来。
        auto k_active = k_total->slice(0, 0, total_len);
        auto v_active = v_total->slice(0, 0, total_len);

        // Self Attention
        auto attn_out = Tensor::create({ntoken, meta.nh, meta.dh}, dtype, LLAISYS_DEVICE_CPU);
        float scale = 1.0f / sqrtf((float)meta.dh);
        ops::self_attention(attn_out, q, k_active, v_active, scale);
        
        // O Proj
        auto attn_out_view = attn_out->view({ntoken, meta.nh * meta.dh});
        auto o_out = Tensor::create(x->shape(), dtype, LLAISYS_DEVICE_CPU);
        ops::linear(o_out, attn_out_view, model->get_tensor(w.attn_o_w[i]), nullptr);
        
        // Residual Add 1: x = residual + o_out
        // 手动实现 add 循环，或假设有 ops::add
        // 这里为了作业完整性，模拟 ops::add
        {
             // 简单 float add 模拟
             float* dst = (float*)x->data();
             float* src1 = (float*)residual->data();
             float* src2 = (float*)o_out->data();
             for(size_t j=0; j<x->numel(); ++j) dst[j] = src1[j] + src2[j];
        }

        residual = x; // 更新残差基准

        // --- MLP Block ---
        // RMS Norm
        ops::rms_norm(norm_out, x, model->get_tensor(w.mlp_norm_w[i]), meta.epsilon);
        
        // Gate & Up
        auto gate = Tensor::create({ntoken, meta.di}, dtype, LLAISYS_DEVICE_CPU);
        auto up = Tensor::create({ntoken, meta.di}, dtype, LLAISYS_DEVICE_CPU);
        ops::linear(gate, norm_out, model->get_tensor(w.mlp_gate_w[i]), nullptr);
        ops::linear(up, norm_out, model->get_tensor(w.mlp_up_w[i]), nullptr);
        
        // SwiGLU
        auto mlp_act = Tensor::create({ntoken, meta.di}, dtype, LLAISYS_DEVICE_CPU);
        ops::swiglu(mlp_act, gate, up);
        
        // Down
        auto mlp_out = Tensor::create(x->shape(), dtype, LLAISYS_DEVICE_CPU);
        ops::linear(mlp_out, mlp_act, model->get_tensor(w.mlp_down_w[i]), nullptr);
        
        // Residual Add 2
        {
             float* dst = (float*)x->data();
             float* src1 = (float*)residual->data();
             float* src2 = (float*)mlp_out->data();
             for(size_t j=0; j<x->numel(); ++j) dst[j] = src1[j] + src2[j];
        }
    } // End Layers Loop

    // 3. Final Norm
    auto final_norm = Tensor::create(x->shape(), dtype, LLAISYS_DEVICE_CPU);
    ops::rms_norm(final_norm, x, model->get_tensor(w.out_norm_w), meta.epsilon);

    // 4. LM Head (只需要最后一个 token)
    // Slice last row: [1, hidden]
    auto last_hidden = final_norm->slice(0, ntoken - 1, ntoken);
    auto logits = Tensor::create({1, meta.voc}, dtype, LLAISYS_DEVICE_CPU);
    ops::linear(logits, last_hidden, model->get_tensor(w.out_embed), nullptr); // lm_head weight 通常复用或独立，Qwen2 独立？需检查 keys
    // 注意：DeepSeek/Qwen 的 lm_head 通常不共享 embedding，需确认 load 逻辑
    
    // 5. Argmax
    auto max_val = Tensor::create({1}, dtype, LLAISYS_DEVICE_CPU);
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
    ops::argmax(max_idx, max_val, logits);
    
    int64_t next_token = ((int64_t*)max_idx->data())[0];
    
    // 更新全局位置
    model->current_pos += ntoken;
    
    return next_token;
}

}
