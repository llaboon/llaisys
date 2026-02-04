#include "llaisys/models/qwen2.h"
#include "llaisys/tensor.h"
#include "../ops/op.hpp" // 引用之前的算子
#include "../../utils.hpp" // 引用 cast 工具
#include <vector>
#include <iostream>
#include <cstring>
#include <cmath>

using namespace llaisys;

// ==========================================
// Helper: Tensor Addition (Residual Connection)
// ==========================================
template <typename T>
void add_kernel(std::byte* dst_bytes, const std::byte* src_bytes, size_t numel) {
    T* dst = reinterpret_cast<T*>(dst_bytes);
    const T* src = reinterpret_cast<const T*>(src_bytes);
    for (size_t i = 0; i < numel; ++i) {
        float a = utils::cast<float>(dst[i]);
        float b = utils::cast<float>(src[i]);
        dst[i] = utils::cast<T>(a + b);
    }
}

void tensor_add(tensor_t dst, tensor_t src) {
    if (dst->dtype() != src->dtype() || dst->numel() != src->numel()) {
        std::cerr << "Tensor Add mismatch!" << std::endl;
        return;
    }
    switch (dst->dtype()) {
        case LLAISYS_DTYPE_F32:
            add_kernel<float>(dst->data(), src->data(), dst->numel());
            break;
        case LLAISYS_DTYPE_F16:
            add_kernel<fp16_t>(dst->data(), src->data(), dst->numel());
            break;
        case LLAISYS_DTYPE_BF16:
            add_kernel<bf16_t>(dst->data(), src->data(), dst->numel());
            break;
        default:
            break;
    }
}

// ==========================================
// Qwen2 Model Structure
// ==========================================
struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    
    // KV Cache: [layer]
    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;
    
    int64_t current_pos = 0;
    
    LlaisysQwen2Model(const LlaisysQwen2Meta *m) : meta(*m) {
        // 分配权重指针数组 (初始化为 nullptr)
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
    }
    
    // 安全地从 void* 恢复 shared_ptr<Tensor>
    tensor_t get_tensor(llaisysTensor_t t) {
        if (!t) return nullptr;
        // 假设 Python 传递的是指向 shared_ptr 的指针
        return *reinterpret_cast<tensor_t*>(t);
    }
};

// ==========================================
// C API Implementation
// ==========================================
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
    auto input = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
    input->load(token_ids); 

    // 获取 Embedding 权重以确定数据类型 (FP32/BF16)
    auto embed_w = model->get_tensor(w.in_embed);
    auto dtype = embed_w->dtype();
    
    // 懒加载初始化 KV Cache
    if (model->k_cache.empty()) {
        for(size_t i=0; i<meta.nlayer; ++i) {
            // Cache Shape: [max_seq, nkvh, head_dim]
            model->k_cache.push_back(Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, dtype, LLAISYS_DEVICE_CPU));
            model->v_cache.push_back(Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, dtype, LLAISYS_DEVICE_CPU));
        }
    }

    // 1. Embedding
    auto x = Tensor::create({ntoken, meta.hs}, dtype, LLAISYS_DEVICE_CPU);
    ops::embedding(x, input, embed_w);

    // 2. Layers Loop
    for (size_t i = 0; i < meta.nlayer; ++i) {
        auto residual = x; // 保存残差输入
        
        // --- Attention Block ---
        // RMS Norm
        auto norm_out = Tensor::create(x->shape(), dtype, LLAISYS_DEVICE_CPU);
        ops::rms_norm(norm_out, x, model->get_tensor(w.attn_norm_w[i]), meta.epsilon);
        
        // QKV Projection
        auto q = Tensor::create({ntoken, meta.nh, meta.dh}, dtype, LLAISYS_DEVICE_CPU);
        auto k = Tensor::create({ntoken, meta.nkvh, meta.dh}, dtype, LLAISYS_DEVICE_CPU);
        auto v = Tensor::create({ntoken, meta.nkvh, meta.dh}, dtype, LLAISYS_DEVICE_CPU);
        
        // View for Linear: [ntoken, hidden]
        auto q_flat = q->view({ntoken, meta.nh * meta.dh});
        auto k_flat = k->view({ntoken, meta.nkvh * meta.dh});
        auto v_flat = v->view({ntoken, meta.nkvh * meta.dh});
        
        ops::linear(q_flat, norm_out, model->get_tensor(w.attn_q_w[i]), model->get_tensor(w.attn_q_b[i]));
        ops::linear(k_flat, norm_out, model->get_tensor(w.attn_k_w[i]), model->get_tensor(w.attn_k_b[i]));
        ops::linear(v_flat, norm_out, model->get_tensor(w.attn_v_w[i]), model->get_tensor(w.attn_v_b[i]));
        
        // RoPE
        auto pos_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
        std::vector<int64_t> pos_vec(ntoken);
        for(size_t p=0; p<ntoken; ++p) pos_vec[p] = model->current_pos + p;
        pos_ids->load(pos_vec.data());
        
        ops::rope(q, q, pos_ids, meta.theta);
        ops::rope(k, k, pos_ids, meta.theta);
        
        // KV Cache Update
        // 计算偏移量: current_pos * nkvh * dh * element_size
        size_t element_size = q->elementSize();
        size_t offset_bytes = model->current_pos * meta.nkvh * meta.dh * element_size;
        size_t copy_bytes = ntoken * meta.nkvh * meta.dh * element_size;
        
        // 直接内存拷贝到 Cache Tensor 的对应位置
        std::byte* k_dst = model->k_cache[i]->data() + offset_bytes;
        std::byte* v_dst = model->v_cache[i]->data() + offset_bytes;
        
        std::memcpy(k_dst, k->data(), copy_bytes);
        std::memcpy(v_dst, v->data(), copy_bytes);
        
        // Prepare K, V for Attention (View of Cache)
        // 从 Cache 中切片出 [0, current_pos + ntoken]
        size_t total_len = model->current_pos + ntoken;
        auto k_active = model->k_cache[i]->slice(0, 0, total_len);
        auto v_active = model->v_cache[i]->slice(0, 0, total_len);

        // Self Attention
        auto attn_out = Tensor::create({ntoken, meta.nh, meta.dh}, dtype, LLAISYS_DEVICE_CPU);
        float scale = 1.0f / sqrtf((float)meta.dh);
        ops::self_attention(attn_out, q, k_active, v_active, scale);
        
        // Output Projection
        auto attn_out_flat = attn_out->view({ntoken, meta.nh * meta.dh});
        auto o_out = Tensor::create(x->shape(), dtype, LLAISYS_DEVICE_CPU);
        ops::linear(o_out, attn_out_flat, model->get_tensor(w.attn_o_w[i]), nullptr);
        
        // Residual Add 1: x = x + o_out
        tensor_add(x, o_out);
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
        
        // Residual Add 2: x = x + mlp_out
        tensor_add(x, mlp_out);
    } // End Layers Loop

    // 3. Final Norm
    auto final_norm = Tensor::create(x->shape(), dtype, LLAISYS_DEVICE_CPU);
    ops::rms_norm(final_norm, x, model->get_tensor(w.out_norm_w), meta.epsilon);

    // 4. LM Head
    // 只取最后一个 token: [1, hidden]
    auto last_hidden = final_norm->slice(0, ntoken - 1, ntoken);
    auto logits = Tensor::create({1, meta.voc}, dtype, LLAISYS_DEVICE_CPU);
    ops::linear(logits, last_hidden, model->get_tensor(w.out_embed), nullptr);
    
    // 5. Argmax
    auto max_val = Tensor::create({1}, dtype, LLAISYS_DEVICE_CPU);
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
    ops::argmax(max_idx, max_val, logits);
    
    int64_t next_token = ((int64_t*)max_idx->data())[0];
    
    // 更新全局位置
    model->current_pos += ntoken;
    
    return next_token;
}

} // extern "C"
