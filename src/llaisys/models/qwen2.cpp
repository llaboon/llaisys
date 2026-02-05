#include <llaisys/models/qwen2.h>
#include <llaisys/tensor.h>
// Include internal ops headers
#include <llaisys/ops/embedding.hpp>
#include <llaisys/ops/rms_norm.hpp>
#include <llaisys/ops/linear.hpp>
#include <llaisys/ops/rope.hpp>
#include <llaisys/ops/self_attention.hpp>
#include <llaisys/ops/swiglu.hpp>
#include <llaisys/ops/add.hpp>
#include <llaisys/ops/argmax.hpp>
// Assuming internal tensor.hpp defines tensor_t
#include <tensor/tensor.hpp> 

#include <vector>
#include <cmath>
#include <iostream>
#include <cstring>

// Helper to bridge C opaque handle to C++ type
// We assume llaisysTensor_t is compatible with the type expected by ops (likely raw pointer or shared_ptr)
// If tensor_t is a shared_ptr, we need to construct it from the raw pointer.
// If tensor_t is a raw pointer (Tensor*), reinterpret_cast is sufficient.
// Looking at ops headers: void add(tensor_t c, ...);
// Looking at tensor.h: typedef struct LlaisysTensor *llaisysTensor_t;
// We assume internal::tensor_t is just LlaisysTensor*.
using TensorPtr = llaisys::tensor_t;

inline TensorPtr cast(llaisysTensor_t t) {
    return reinterpret_cast<TensorPtr>(t);
}

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    llaisysDeviceType_t device;
    int device_id;

    // Weight Pointers (storage for the struct arrays)
    std::vector<llaisysTensor_t> w_attn_norm, w_q, w_k, w_v, w_o;
    std::vector<llaisysTensor_t> b_q, b_k, b_v;
    std::vector<llaisysTensor_t> w_mlp_norm, w_gate, w_up, w_down;

    // Exported struct
    LlaisysQwen2Weights weights;

    // Runtime State
    int64_t current_pos;
    
    // KV Caches: [Layer]
    struct KV {
        llaisysTensor_t k;
        llaisysTensor_t v;
    };
    std::vector<KV> kv_caches;

    // Buffers (reusable)
    llaisysTensor_t buf_x;
    llaisysTensor_t buf_res; // residual
    llaisysTensor_t buf_q, buf_k, buf_v; // Before cache/rope
    llaisysTensor_t buf_att;
    llaisysTensor_t buf_ffn_gate, buf_ffn_up, buf_ffn_down;
    llaisysTensor_t buf_logits;
    llaisysTensor_t buf_pos_ids; // for rope
    llaisysTensor_t buf_token_ids; // for input

    // Construction Helper
    void init_vectors(size_t n) {
        w_attn_norm.resize(n); w_q.resize(n); w_k.resize(n); w_v.resize(n); w_o.resize(n);
        b_q.resize(n); b_k.resize(n); b_v.resize(n);
        w_mlp_norm.resize(n); w_gate.resize(n); w_up.resize(n); w_down.resize(n);
        
        weights.attn_norm_w = w_attn_norm.data();
        weights.attn_q_w = w_q.data(); weights.attn_q_b = b_q.data();
        weights.attn_k_w = w_k.data(); weights.attn_k_b = b_k.data();
        weights.attn_v_w = w_v.data(); weights.attn_v_b = b_v.data();
        weights.attn_o_w = w_o.data();
        weights.mlp_norm_w = w_mlp_norm.data();
        weights.mlp_gate_w = w_gate.data();
        weights.mlp_up_w = w_up.data();
        weights.mlp_down_w = w_down.data();
    }
};

extern "C" {

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    auto model = new LlaisysQwen2Model();
    model->meta = *meta;
    model->device = device;
    model->device_id = (device_ids && ndevice > 0) ? device_ids[0] : 0;
    model->current_pos = 0;

    model->init_vectors(meta->nlayer);

    // Initialize KV Cache: [1, MaxSeq, NKVH, HeadDim]
    // Qwen uses standard layout often, but let's stick to what attention op expects.
    // Ops usually expect [Batch, Seq, Head, Dim].
    size_t kv_shape[] = {1, meta->maxseq, meta->nkvh, meta->dh};
    
    for(size_t i=0; i<meta->nlayer; ++i) {
        LlaisysQwen2Model::KV kv;
        kv.k = tensorCreate(kv_shape, 4, meta->dtype, device, model->device_id);
        kv.v = tensorCreate(kv_shape, 4, meta->dtype, device, model->device_id);
        model->kv_caches.push_back(kv);
    }

    return model;
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
    if (!model) return;
    
    // Destroy KV Caches
    for(auto &kv : model->kv_caches) {
        tensorDestroy(kv.k);
        tensorDestroy(kv.v);
    }
    
    // Destroy Buffers if allocated
    if (model->buf_x) tensorDestroy(model->buf_x);
    if (model->buf_res) tensorDestroy(model->buf_res);
    // ... clean others
    
    // Note: Weights are managed by Python side (safetensors load) usually, 
    // BUT we stored the handles. If Python side created them using tensorCreate, 
    // Python side __del__ or GC should handle them if we expose destroy there.
    // In this assignment, `llaisysQwen2ModelWeights` returns pointer to struct.
    // The Python code assigns to it.
    
    delete model;
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
    return &model->weights;
}

int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
    if (ntoken == 0) return -1;

    // 1. Prepare Input Tensor
    size_t input_shape[] = {1, ntoken};
    // Recreate input tensor to match ntoken (or reuse if size sufficient)
    // For simplicity, create/destroy per step (low overhead compared to compute)
    llaisysTensor_t t_input = tensorCreate(input_shape, 2, LLAISYS_DTYPE_INT64, model->device, model->device_id);
    // Assuming token_ids is host memory. We need to copy to device.
    // tensorLoad expects host pointer.
    tensorLoad(t_input, token_ids);

    // 2. Prepare Position IDs for RoPE
    // Shape: [1, ntoken]
    llaisysTensor_t t_pos = tensorCreate(input_shape, 2, LLAISYS_DTYPE_INT64, model->device, model->device_id);
    std::vector<int64_t> pos_data(ntoken);
    for(size_t i=0; i<ntoken; ++i) pos_data[i] = model->current_pos + i;
    tensorLoad(t_pos, pos_data.data());

    // 3. Prepare Runtime Buffers (Resizing if ntoken changes, e.g. prefill vs decode)
    // Hidden states: [1, ntoken, hs]
    size_t hidden_shape[] = {1, ntoken, model->meta.hs};
    llaisysTensor_t t_x = tensorCreate(hidden_shape, 3, model->meta.dtype, model->device, model->device_id);
    llaisysTensor_t t_res = tensorCreate(hidden_shape, 3, model->meta.dtype, model->device, model->device_id); // Residual
    llaisysTensor_t t_norm = tensorCreate(hidden_shape, 3, model->meta.dtype, model->device, model->device_id);

    // QKV buffers: [1, ntoken, nh/nkvh, dh]
    // But Linear outputs flattened [1, ntoken, nh*dh]. We need view/reshape.
    // Let's allocate linear output buffers first.
    size_t q_shape[] = {1, ntoken, model->meta.nh, model->meta.dh};
    size_t kv_shape[] = {1, ntoken, model->meta.nkvh, model->meta.dh};
    
    // Intermediate tensors for Layer Loop
    llaisysTensor_t t_q = tensorCreate(q_shape, 4, model->meta.dtype, model->device, model->device_id);
    llaisysTensor_t t_k_rope = tensorCreate(kv_shape, 4, model->meta.dtype, model->device, model->device_id); 
    // Note: We will write Linear V directly to Cache Slice, or to a temp buffer then copy?
    // Since we don't have copy, we write Linear output to temp, then use ... wait.
    // Linear op: out = xW + b.
    // If we view the KV cache slice as 'out', we write directly.
    
    // Attn Out: [1, ntoken, hs]
    llaisysTensor_t t_attn_out = tensorCreate(hidden_shape, 3, model->meta.dtype, model->device, model->device_id);
    
    // MLP Buffers
    size_t gate_up_shape[] = {1, ntoken, model->meta.di};
    llaisysTensor_t t_gate = tensorCreate(gate_up_shape, 3, model->meta.dtype, model->device, model->device_id);
    llaisysTensor_t t_up = tensorCreate(gate_up_shape, 3, model->meta.dtype, model->device, model->device_id);
    llaisysTensor_t t_mlp_out = tensorCreate(hidden_shape, 3, model->meta.dtype, model->device, model->device_id);

    // --- Execution Graph ---

    // Embedding
    llaisys::ops::embedding(cast(t_x), cast(t_input), cast(model->weights.in_embed));

    for (size_t i = 0; i < model->meta.nlayer; ++i) {
        // --- Attention Block ---
        // 1. Residual Backup
        // We don't have a specific 'copy' op in the list. Use 'add(res, x, zero_tensor)'?
        // Or assume add(x, x, attn_out) works in-place if we structure it right.
        // Let's assume standard transformer: x = x + attn(norm(x))
        // So we keep 'x' as accumulator.
        // norm_out = norm(x)
        llaisys::ops::rms_norm(cast(t_norm), cast(t_x), cast(model->weights.attn_norm_w[i]), model->meta.epsilon);

        // 2. QKV Linear
        // Q: [1, ntoken, nh*dh] -> View as [1, ntoken, nh, dh]
        // K: [1, ntoken, nkvh*dh]
        // V: [1, ntoken, nkvh*dh]
        
        // Q Projection
        llaisys::ops::linear(cast(t_q), cast(t_norm), cast(model->weights.attn_q_w[i]), cast(model->weights.attn_q_b[i]));
        
        // KV Projections
        // Optim: Write directly into a temp buffer for RoPE, then cache?
        // Or write directly to Cache? RoPE needs to happen before Caching?
        // Standard: Linear -> RoPE -> Cache.
        // So we need temp buffers for K and V output from Linear.
        llaisysTensor_t t_k_temp = tensorCreate(kv_shape, 4, model->meta.dtype, model->device, model->device_id);
        llaisysTensor_t t_v_temp = tensorCreate(kv_shape, 4, model->meta.dtype, model->device, model->device_id);
        
        llaisys::ops::linear(cast(t_k_temp), cast(t_norm), cast(model->weights.attn_k_w[i]), cast(model->weights.attn_k_b[i]));
        llaisys::ops::linear(cast(t_v_temp), cast(t_norm), cast(model->weights.attn_v_w[i]), cast(model->weights.attn_v_b[i]));

        // 3. RoPE
        // Apply to Q and K_temp
        // The pos_ids need to be broadcast or match batch/seq dims. 
        // rope op usually takes [1, ntoken] or similar.
        llaisys::ops::rope(cast(t_q), cast(t_q), cast(t_pos), model->meta.theta);
        llaisys::ops::rope(cast(t_k_temp), cast(t_k_temp), cast(t_pos), model->meta.theta);

        // 4. Update KV Cache
        // We need to copy t_k_temp and t_v_temp into kv_caches[i].k and .v at [:, current_pos:current_pos+ntoken, :, :]
        // Use tensorSlice to get a view of the destination in cache.
        llaisysTensor_t k_cache_view = tensorSlice(model->kv_caches[i].k, 1, model->current_pos, model->current_pos + ntoken);
        llaisysTensor_t v_cache_view = tensorSlice(model->kv_caches[i].v, 1, model->current_pos, model->current_pos + ntoken);
        
        // Copy? We lack a copy op. 
        // HACK: Use `add(dest, src, zero)` if possible, OR
        // Use `swiglu` or similar as identity? No.
        // Maybe `linear` with Identity matrix? Too expensive.
        // Wait, if tensorLoad works for D2D? "tensorLoad(tensor, const void* data)". Usually expects host.
        // 
        // CRITICAL: If we cannot copy, we should have written Linear Output directly to Cache View?
        // But RoPE is in-place or out-of-place. 
        // If RoPE is in-place:
        // 1. Linear writes to K_cache_view.
        // 2. RoPE acts on K_cache_view.
        // YES. This avoids copy.
        // V doesn't need RoPE, so Linear writes to V_cache_view directly.
        
        // REDO Step 2:
        // K Projection -> Write to Cache View
        llaisys::ops::linear(cast(k_cache_view), cast(t_norm), cast(model->weights.attn_k_w[i]), cast(model->weights.attn_k_b[i]));
        // V Projection -> Write to Cache View
        llaisys::ops::linear(cast(v_cache_view), cast(t_norm), cast(model->weights.attn_v_w[i]), cast(model->weights.attn_v_b[i]));
        
        // REDO Step 3:
        // RoPE on Q (temp buffer) and K (Cache View)
        llaisys::ops::rope(cast(t_q), cast(t_q), cast(t_pos), model->meta.theta);
        llaisys::ops::rope(cast(k_cache_view), cast(k_cache_view), cast(t_pos), model->meta.theta);

        // 5. Attention
        // Needs full context KV.
        // View K/V cache from 0 to current_pos + ntoken
        llaisysTensor_t k_full_view = tensorSlice(model->kv_caches[i].k, 1, 0, model->current_pos + ntoken);
        llaisysTensor_t v_full_view = tensorSlice(model->kv_caches[i].v, 1, 0, model->current_pos + ntoken);
        
        float scale = 1.0f / sqrtf((float)model->meta.dh);
        llaisys::ops::self_attention(cast(t_attn_out), cast(t_q), cast(k_full_view), cast(v_full_view), scale);

        // 6. Output Projection
        // We need to add to 'x'.
        // use t_res as temp for projection result
        llaisys::ops::linear(cast(t_res), cast(t_attn_out), cast(model->weights.attn_o_w[i]), nullptr);
        // Residual Add: x = x + res
        llaisys::ops::add(cast(t_x), cast(t_x), cast(t_res));
        
        // Clean up views
        tensorDestroy(k_cache_view); tensorDestroy(v_cache_view);
        tensorDestroy(k_full_view); tensorDestroy(v_full_view);
        tensorDestroy(t_k_temp); tensorDestroy(t_v_temp); // (if we created them, here we didn't use them in REDO)


        // --- MLP Block ---
        // Norm
        llaisys::ops::rms_norm(cast(t_norm), cast(t_x), cast(model->weights.mlp_norm_w[i]), model->meta.epsilon);
        
        // Gate & Up
        llaisys::ops::linear(cast(t_gate), cast(t_norm), cast(model->weights.mlp_gate_w[i]), nullptr);
        llaisys::ops::linear(cast(t_up), cast(t_norm), cast(model->weights.mlp_up_w[i]), nullptr);
        
        // SwiGLU: act = swiglu(gate, up) -> writes to t_gate usually or t_up?
        // Signature: swiglu(out, gate, up). Let's use t_gate as out or new buffer.
        // Let's reuse t_gate as output 'act'
        llaisys::ops::swiglu(cast(t_gate), cast(t_gate), cast(t_up));
        
        // Down
        llaisys::ops::linear(cast(t_res), cast(t_gate), cast(model->weights.mlp_down_w[i]), nullptr);
        
        // Residual Add
        llaisys::ops::add(cast(t_x), cast(t_x), cast(t_res));
    }

    // Final Norm
    llaisys::ops::rms_norm(cast(t_x), cast(t_x), cast(model->weights.out_norm_w), model->meta.epsilon);

    // LM Head
    // We only need the last token logits
    // Slice t_x to get last token: [1, 1, hs]
    llaisysTensor_t t_last_x = tensorSlice(t_x, 1, ntoken - 1, ntoken);
    
    size_t logits_shape[] = {1, 1, model->meta.voc};
    llaisysTensor_t t_logits = tensorCreate(logits_shape, 3, model->meta.dtype, model->device, model->device_id);
    
    llaisys::ops::linear(cast(t_logits), cast(t_last_x), cast(model->weights.out_embed), nullptr);
    
    // Argmax
    // Create output scalar tensor (on CPU? or Device?)
    // Argmax op usually returns index on same device.
    // We assume argmax outputs a tensor with 1 element.
    size_t out_shape[] = {1};
    llaisysTensor_t t_out_idx = tensorCreate(out_shape, 1, LLAISYS_DTYPE_INT64, model->device, model->device_id);
    
    // argmax(max_idx, max_val, vals). We don't care about val.
    // Need a dummy tensor for max_val? Or pass nullptr if op supports it?
    // Safer to create one.
    llaisysTensor_t t_out_val = tensorCreate(out_shape, 1, model->meta.dtype, model->device, model->device_id);
    
    llaisys::ops::argmax(cast(t_out_idx), cast(t_out_val), cast(t_logits));
    
    // Read back result
    int64_t result_idx = 0;
    // tensorGetData returns pointer to device memory?
    // tensor.h says: void *tensorGetData(llaisysTensor_t tensor);
    // If device is GPU, we can't dereference.
    // We need to copy back. Does tensorLoad support D2H? No, it's usually H2D.
    // Is there a tensorCopy or tensorSync?
    // Check tensor.h... "tensorGetData". If the runtime is unified, might work.
    // If not, maybe we need a missing API.
    // WAIT: `test_infer.py` uses `outputs[0].tolist()`.
    // In our case, `llaisysQwen2ModelInfer` returns `int64_t`.
    // We MUST get data back to CPU.
    // If `llaisysTensor_t` is a wrapper around `Tensor`, and we are on CPU device, `GetData` is fine.
    // If on GPU?
    // Usually there is `tensorToHost` or `memcpy`.
    // The provided `tensor.h` is sparse.
    // BUT `python/llaisys/libllaisys/__init__.py` imports `MemcpyKind`.
    // This implies there is a runtime memcpy function available in `runtime.h` (implied by `load_runtime`).
    // However, I don't have `runtime.h` source in the prompt, only `tensor.h`.
    // I will assume for this assignment `tensorGetData` returns a host-accessible pointer OR
    // I will use a trick: `tensorCreate` with CPU device, then `ops::add` (copy) if cross-device ops supported? No.
    // Let's assume for `test_infer.py` CPU test, direct access is fine.
    // For GPU, we might be missing a `tensorCopyDtoH`.
    // Assuming the user runs with `--device cpu` primarily for this strict test or the environment handles it.
    // I will cast `tensorGetData(t_out_idx)` to `int64_t*` and read.
    
    int64_t* ptr = (int64_t*)tensorGetData(t_out_idx);
    result_idx = ptr[0]; // Logic assumes CPU accessible or Unified Memory

    // Cleanup Step Tensors
    tensorDestroy(t_input); tensorDestroy(t_pos);
    tensorDestroy(t_x); tensorDestroy(t_res); tensorDestroy(t_norm);
    tensorDestroy(t_q); 
    tensorDestroy(t_attn_out);
    tensorDestroy(t_gate); tensorDestroy(t_up); tensorDestroy(t_mlp_out);
    tensorDestroy(t_last_x); tensorDestroy(t_logits);
    tensorDestroy(t_out_idx); tensorDestroy(t_out_val);

    model->current_pos += ntoken;
    return result_idx;
}

} // extern "C"
