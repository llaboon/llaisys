// include/llaisys/models/qwen2.h
#pragma once

#include "llaisys/llaisys.h" // 或者是 tensor.h

#ifdef __cplusplus
extern "C" {
#endif

// =========================================================
// 必须修改这里：全部改成 void* 和 void**
// 这样才能和 .cc 文件里的 new void*[] 匹配
// 同时也解决了 Python ctypes 内存布局不对齐的问题
// =========================================================
typedef struct LlaisysQwen2Weights {
    void* in_embed;       // 之前是 llaisysTensor_t
    void* out_embed;
    void* out_norm_w;
    
    // 指针数组，指向 void*
    void** attn_norm_w;   // 之前是 llaisysTensor_t*
    void** attn_q_w;
    void** attn_q_b;
    void** attn_k_w;
    void** attn_k_b;
    void** attn_v_w;
    void** attn_v_b;
    void** attn_o_w;
    void** mlp_norm_w;
    void** mlp_gate_w;
    void** mlp_up_w;
    void** mlp_down_w;
} LlaisysQwen2Weights;

typedef struct LlaisysQwen2Meta {
    size_t nlayer;
    size_t hs;
    size_t nh;
    size_t nkvh;
    size_t dh;
    size_t di;
    size_t maxseq;
    size_t voc;
    float epsilon;
    float theta;
    int64_t end_token;
    int dtype;
} LlaisysQwen2Meta;

typedef struct LlaisysQwen2Model LlaisysQwen2Model;

LLAISYS_EXPORT LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);
LLAISYS_EXPORT void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model);
LLAISYS_EXPORT LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model);
LLAISYS_EXPORT int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken);

#ifdef __cplusplus
}
#endif
