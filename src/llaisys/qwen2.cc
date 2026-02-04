#include "llaisys/models/qwen2.h"
#include "llaisys/tensor.h"
#include "../utils.hpp" // 引用 src/utils.hpp
#include <vector>
#include <iostream>
#include <cstring>
#include <cmath>

using namespace llaisys;

// ==========================================
// 1. 手动声明算子 (Forward Declarations)
// ==========================================
// 因为 src/ops/ 下没有统一的 op.hpp，我们需要手动告诉编译器这些函数的存在
namespace llaisys::ops {
    void embedding(tensor_t out, tensor_t index, tensor_t weight);
    void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps);
    void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);
    void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);
    void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
    void swiglu(tensor_t out, tensor_t gate, tensor_t up);
    void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
}

// ==========================================
// 2. 辅助函数: Tensor 加法 (用于残差连接)
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
// 3. Qwen2 模型实现
// ==========================================
struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    
    // KV Cache
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
    }
    
    tensor_t get_tensor(llaisysTensor_t t) {
        if (!t) return nullptr;
        return *reinterpret_cast<tensor_t*>(t);
    }
};

// ==========================================
// 4. C API 导出实现
// ==========================================
extern "C" {

#if defined(_WIN32)
#define LLAISYS_EXPORT __declspec(dllexport)
#else
#define LLAISYS_EXPORT __attribute__((visibility("default")))
#endif

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
    
    // 0. 准备输入
    auto input = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
    input->load(token_ids); 

    auto embed_w = model->get_tensor(w.in_embed);
    auto dtype = embed_w->dtype();
    
    // 初始化 KV Cache
    if (model->k_cache.empty()) {
        for(size_t i=0; i<meta.nlayer; ++i) {
            model->k_cache.push_back(Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, dtype, LLAISYS_DEVICE_CPU));
            model->v_cache.push_back(Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, dtype, LLAISYS_DEVICE_CPU));
        }
    }

    // 1. Embedding
    auto x = Tensor::create({ntoken, meta.hs}, dtype, LLAISYS_DEVICE_CPU);
    ops::embedding(x, input, embed_w);

    // 2. Transformer 层循环
    for (size_t i = 0; i < meta.nlayer; ++i) {
        auto residual = x;
        
        // --- Attention 模块 ---
        auto norm_out = Tensor::create(x->shape(), dtype, LLAISYS_DEVICE_CPU);
        ops::rms_norm(norm_out, x, model->get_tensor(w.attn_norm_w[i]), meta.epsilon);
        
        auto q = Tensor::create({ntoken, meta.nh, meta.dh}, dtype, LLAISYS_DEVICE_CPU);
        auto k = Tensor::create({ntoken, meta.nkvh, meta.dh}, dtype, LLAISYS_DEVICE_CPU);
        auto v = Tensor::create({ntoken, meta.nkvh, meta.dh}, dtype, LLAISYS_DEVICE_CPU);
        
        auto q_view = q->view({ntoken, meta.nh * meta.dh});
        auto k_view = k->view({ntoken, meta.nkvh * meta.dh});
        auto v_view = v->view({ntoken, meta.nkvh * meta.dh});
        
        ops::linear(q_view, norm_out, model->get_tensor(w.attn_q_w[i]), model->get_tensor(w.attn_q_b[i]));
        ops::linear(k_view, norm_out, model->get_tensor(w.attn_k_w[i]), model->get_tensor(w.attn_k_b[i]));
        ops::linear(v_view, norm_out, model->get_tensor(w.attn_v_w[i]), model->get_tensor(w.attn_v_b[i]));
        
        // RoPE
        auto pos_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
        std::vector<int64_t> pos_vec(ntoken);
        for(size_t p=0; p<ntoken; ++p) pos_vec[p] = model->current_pos + p;
        pos_ids->load(pos_vec.data());
        
        ops::rope(q, q, pos_ids, meta.theta);
        ops::rope(k, k, pos_ids, meta.theta);
        
        // KV Cache 更新 (Copy)
        size_t element_size = q->elementSize();
        size_t offset_bytes = model->current_pos * meta.nkvh * meta.dh * element_size;
        size_t copy_bytes = ntoken * meta.nkvh * meta.dh * element_size;
        
        std::memcpy(model->k_cache[i]->data() + offset_bytes, k->data(), copy_bytes);
        std::memcpy(model->v_cache[i]->data() + offset_bytes, v->data(), copy_bytes);
        
        // Self Attention
        size_t total_len = model->current_pos + ntoken;
        auto k_active = model->k_cache[i]->slice(0, 0, total_len);
        auto v_active = model->v_cache[i]->slice(0, 0, total_len);

        auto attn_out = Tensor::create({ntoken, meta.nh, meta.dh}, dtype, LLAISYS_DEVICE_CPU);
        float scale = 1.0f / sqrtf((float)meta.dh);
        ops::self_attention(attn_out, q, k_active, v_active, scale);
        
        auto attn_out_view = attn_out->view({ntoken, meta.nh * meta.dh});
        auto o_out = Tensor::create(x->shape(), dtype, LLAISYS_DEVICE_CPU);
        ops::linear(o_out, attn_out_view, model->get_tensor(w.attn_o_w[i]), nullptr);
        
        tensor_add(x, o_out);
        residual = x;

        // --- MLP 模块 ---
        ops::rms_norm(norm_out, x, model->get_tensor(w.mlp_norm_w[i]), meta.epsilon);
        
        auto gate = Tensor::create({ntoken, meta.di}, dtype, LLAISYS_DEVICE_CPU);
        auto up = Tensor::create({ntoken, meta.di}, dtype, LLAISYS_DEVICE_CPU);
        ops::linear(gate, norm_out, model->get_tensor(w.mlp_gate_w[i]), nullptr);
        ops::linear(up, norm_out, model->get_tensor(w.mlp_up_w[i]), nullptr);
        
        auto mlp_act = Tensor::create({ntoken, meta.di}, dtype, LLAISYS_DEVICE_CPU);
        ops::swiglu(mlp_act, gate, up);
        
        auto mlp_out = Tensor::create(x->shape(), dtype, LLAISYS_DEVICE_CPU);
        ops::linear(mlp_out, mlp_act, model->get_tensor(w.mlp_down_w[i]), nullptr);
        
        tensor_add(x, mlp_out);
    }

    // 3. 最终层归一化
    auto final_norm = Tensor::create(x->shape(), dtype, LLAISYS_DEVICE_CPU);
    ops::rms_norm(final_norm, x, model->get_tensor(w.out_norm_w), meta.epsilon);

    // 4. LM Head (只计算最后一个 Token)
    auto last_hidden = final_norm->slice(0, ntoken - 1, ntoken);
    auto logits = Tensor::create({1, meta.voc}, dtype, LLAISYS_DEVICE_CPU);
    ops::linear(logits, last_hidden, model->get_tensor(w.out_embed), nullptr);
    
    // 5. Argmax 采样
    auto max_val = Tensor::create({1}, dtype, LLAISYS_DEVICE_CPU);
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
    ops::argmax(max_idx, max_val, logits);
    
    int64_t next_token = ((int64_t*)max_idx->data())[0];
    
    // 更新位置
    model->current_pos += ntoken;
    return next_token;
}

} // extern "C"
