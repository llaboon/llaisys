// src/llaisys/models/qwen2.cpp

// 1. 包含公开头文件 (基于 include 目录)
#include <llaisys/models/qwen2.h>
#include <llaisys/tensor.h>

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream> // Debug usage

// 2. 内部类型和函数声明 (Forward Declaration)
// 我们直接声明 Assignment-2 中的 C++ 函数，避免 include 路径猜测错误
// 假设内部 tensor_t 是指向 LlaisysTensor 的指针
namespace llaisys {
    // 假设 internal tensor type 定义
    typedef struct LlaisysTensor *tensor_t; 
    
    namespace ops {
        void embedding(tensor_t out, tensor_t index, tensor_t weight);
        void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps);
        void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);
        void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);
        void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
        void swiglu(tensor_t out, tensor_t gate, tensor_t up);
        void add(tensor_t c, tensor_t a, tensor_t b);
        void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
    }
}

// 3. 辅助函数：类型转换
// 由于 llaisysTensor_t 在 tensor.h 中定义为 struct LlaisysTensor *
// 而上面的 llaisys::tensor_t 也是 struct LlaisysTensor *
// 我们可以直接转换
using InternalTensor = llaisys::tensor_t;
inline InternalTensor cast(llaisysTensor_t t) { return reinterpret_cast<InternalTensor>(t); }


// 4. 模型结构体定义
struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    llaisysDeviceType_t device;
    int device_id;
    int64_t current_pos;

    // Weight Storage (Vectors to hold the pointers)
    std::vector<llaisysTensor_t> v_attn_norm, v_q_w, v_q_b, v_k_w, v_k_b, v_v_w, v_v_b, v_o_w;
    std::vector<llaisysTensor_t> v_mlp_norm, v_gate, v_up, v_down;
    
    // Exported struct to Python
    LlaisysQwen2Weights exported;

    // KV Cache
    struct KVLayer {
        llaisysTensor_t k;
        llaisysTensor_t v;
    };
    std::vector<KVLayer> kv_caches;

    LlaisysQwen2Model() : current_pos(0) {}
};

// 5. 实现导出函数
extern "C" {

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    auto model = new LlaisysQwen2Model();
    model->meta = *meta;
    model->device = device;
    model->device_id = (device_ids && ndevice > 0) ? device_ids[0] : 0;

    size_t n = meta->nlayer;
    
    // Resize vectors
    model->v_attn_norm.resize(n);
    model->v_q_w.resize(n); model->v_q_b.resize(n);
    model->v_k_w.resize(n); model->v_k_b.resize(n);
    model->v_v_w.resize(n); model->v_v_b.resize(n);
    model->v_o_w.resize(n);
    model->v_mlp_norm.resize(n);
    model->v_gate.resize(n); model->v_up.resize(n); model->v_down.resize(n);

    // Link exported struct pointers
    model->exported.attn_norm_w = model->v_attn_norm.data();
    model->exported.attn_q_w = model->v_q_w.data(); model->exported.attn_q_b = model->v_q_b.data();
    model->exported.attn_k_w = model->v_k_w.data(); model->exported.attn_k_b = model->v_k_b.data();
    model->exported.attn_v_w = model->v_v_w.data(); model->exported.attn_v_b = model->v_v_b.data();
    model->exported.attn_o_w = model->v_o_w.data();
    model->exported.mlp_norm_w = model->v_mlp_norm.data();
    model->exported.mlp_gate_w = model->v_gate.data();
    model->exported.mlp_up_w = model->v_up.data();
    model->exported.mlp_down_w = model->v_down.data();

    // Create KV Cache: [1, MaxSeq, NKVH, HeadDim]
    size_t kv_shape[] = {1, meta->maxseq, meta->nkvh, meta->dh};
    for(size_t i=0; i<n; ++i) {
        LlaisysQwen2Model::KVLayer layer;
        layer.k = tensorCreate(kv_shape, 4, meta->dtype, device, model->device_id);
        layer.v = tensorCreate(kv_shape, 4, meta->dtype, device, model->device_id);
        model->kv_caches.push_back(layer);
    }

    return model;
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
    if(!model) return;
    for(auto &l : model->kv_caches) {
        tensorDestroy(l.k);
        tensorDestroy(l.v);
    }
    delete model;
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
    return &model->exported;
}

int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
    if (ntoken == 0) return -1;

    // 1. Prepare Input
    size_t in_shape[] = {1, ntoken};
    llaisysTensor_t t_input = tensorCreate(in_shape, 2, LLAISYS_DTYPE_INT64, model->device, model->device_id);
    tensorLoad(t_input, token_ids); 

    // 2. Prepare Position IDs
    llaisysTensor_t t_pos = tensorCreate(in_shape, 2, LLAISYS_DTYPE_INT64, model->device, model->device_id);
    std::vector<int64_t> pos_vec(ntoken);
    for(size_t i=0; i<ntoken; ++i) pos_vec[i] = model->current_pos + i;
    tensorLoad(t_pos, pos_vec.data());

    // 3. Hidden State
    size_t h_shape[] = {1, ntoken, model->meta.hs};
    llaisysTensor_t t_x = tensorCreate(h_shape, 3, model->meta.dtype, model->device, model->device_id);
    llaisysTensor_t t_norm = tensorCreate(h_shape, 3, model->meta.dtype, model->device, model->device_id);

    // Embedding
    llaisys::ops::embedding(cast(t_x), cast(t_input), cast(model->exported.in_embed));

    // Shapes
    size_t q_shape[] = {1, ntoken, model->meta.nh, model->meta.dh};
    size_t di_shape[] = {1, ntoken, model->meta.di};

    for(size_t i=0; i<model->meta.nlayer; ++i) {
        // --- Attention ---
        llaisys::ops::rms_norm(cast(t_norm), cast(t_x), cast(model->exported.attn_norm_w[i]), model->meta.epsilon);

        // Q
        llaisysTensor_t t_q = tensorCreate(q_shape, 4, model->meta.dtype, model->device, model->device_id);
        llaisys::ops::linear(cast(t_q), cast(t_norm), cast(model->exported.attn_q_w[i]), cast(model->exported.attn_q_b[i]));

        // KV Cache Update
        llaisysTensor_t k_slice = tensorSlice(model->kv_caches[i].k, 1, model->current_pos, model->current_pos + ntoken);
        llaisysTensor_t v_slice = tensorSlice(model->kv_caches[i].v, 1, model->current_pos, model->current_pos + ntoken);
        
        llaisys::ops::linear(cast(k_slice), cast(t_norm), cast(model->exported.attn_k_w[i]), cast(model->exported.attn_k_b[i]));
        llaisys::ops::linear(cast(v_slice), cast(t_norm), cast(model->exported.attn_v_w[i]), cast(model->exported.attn_v_b[i]));
        
        // RoPE
        llaisys::ops::rope(cast(t_q), cast(t_q), cast(t_pos), model->meta.theta);
        llaisys::ops::rope(cast(k_slice), cast(k_slice), cast(t_pos), model->meta.theta);

        // Attention
        llaisysTensor_t k_full = tensorSlice(model->kv_caches[i].k, 1, 0, model->current_pos + ntoken);
        llaisysTensor_t v_full = tensorSlice(model->kv_caches[i].v, 1, 0, model->current_pos + ntoken);
        
        llaisysTensor_t t_attn_out = tensorCreate(h_shape, 3, model->meta.dtype, model->device, model->device_id);
        float scale = 1.0f / sqrtf((float)model->meta.dh);
        
        llaisys::ops::self_attention(cast(t_attn_out), cast(t_q), cast(k_full), cast(v_full), scale);
        
        // Output Projection
        llaisys::ops::linear(cast(t_norm), cast(t_attn_out), cast(model->exported.attn_o_w[i]), nullptr);
        llaisys::ops::add(cast(t_x), cast(t_x), cast(t_norm));

        tensorDestroy(t_q); tensorDestroy(t_attn_out);
        tensorDestroy(k_slice); tensorDestroy(v_slice);
        tensorDestroy(k_full); tensorDestroy(v_full);

        // --- MLP ---
        llaisys::ops::rms_norm(cast(t_norm), cast(t_x), cast(model->exported.mlp_norm_w[i]), model->meta.epsilon);
        
        llaisysTensor_t t_gate = tensorCreate(di_shape, 3, model->meta.dtype, model->device, model->device_id);
        llaisysTensor_t t_up = tensorCreate(di_shape, 3, model->meta.dtype, model->device, model->device_id);
        
        llaisys::ops::linear(cast(t_gate), cast(t_norm), cast(model->exported.mlp_gate_w[i]), nullptr);
        llaisys::ops::linear(cast(t_up), cast(t_norm), cast(model->exported.mlp_up_w[i]), nullptr);
        
        llaisys::ops::swiglu(cast(t_gate), cast(t_gate), cast(t_up));
        
        // Reuse t_norm for down_proj result
        llaisys::ops::linear(cast(t_norm), cast(t_gate), cast(model->exported.mlp_down_w[i]), nullptr);
        llaisys::ops::add(cast(t_x), cast(t_x), cast(t_norm));
        
        tensorDestroy(t_gate); tensorDestroy(t_up);
    }

    // Final Norm
    llaisys::ops::rms_norm(cast(t_x), cast(t_x), cast(model->exported.out_norm_w), model->meta.epsilon);

    // Last Token
    llaisysTensor_t t_last = tensorSlice(t_x, 1, ntoken - 1, ntoken);
    size_t logits_shape[] = {1, 1, model->meta.voc};
    llaisysTensor_t t_logits = tensorCreate(logits_shape, 3, model->meta.dtype, model->device, model->device_id);
    
    llaisys::ops::linear(cast(t_logits), cast(t_last), cast(model->exported.out_embed), nullptr);

    // Argmax
    size_t out_shape[] = {1};
    llaisysTensor_t t_idx = tensorCreate(out_shape, 1, LLAISYS_DTYPE_INT64, model->device, model->device_id);
    llaisysTensor_t t_val = tensorCreate(out_shape, 1, model->meta.dtype, model->device, model->device_id);
    
    llaisys::ops::argmax(cast(t_idx), cast(t_val), cast(t_logits));

    // Get Result
    int64_t result = 0;
    void* ptr = tensorGetData(t_idx);
    if(ptr) {
        result = *(int64_t*)ptr;
    }

    tensorDestroy(t_input); tensorDestroy(t_pos);
    tensorDestroy(t_x); tensorDestroy(t_norm);
    tensorDestroy(t_last); tensorDestroy(t_logits);
    tensorDestroy(t_idx); tensorDestroy(t_val);

    model->current_pos += ntoken;
    return result;
}

} // extern "C"
