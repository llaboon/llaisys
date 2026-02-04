from typing import Sequence
import json
import numpy as np
import ctypes
import safetensors.numpy
from pathlib import Path

from ..libllaisys import LIB_LLAISYS, DeviceType
from ..libllaisys.models.qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights
from .. import Tensor

class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        
        # 1. Load Config
        with open(model_path / "config.json", "r") as f:
            cfg = json.load(f)
            
        self.meta = LlaisysQwen2Meta()
        self.meta.nlayer = cfg["num_hidden_layers"]
        self.meta.hs = cfg["hidden_size"]
        self.meta.nh = cfg["num_attention_heads"]
        self.meta.nkvh = cfg["num_key_value_heads"]
        self.meta.dh = self.meta.hs // self.meta.nh
        self.meta.di = cfg["intermediate_size"]
        self.meta.maxseq = 2048 # 默认或从config读取
        self.meta.voc = cfg["vocab_size"]
        self.meta.epsilon = cfg["rms_norm_eps"]
        self.meta.theta = cfg.get("rope_theta", 10000.0)
        self.meta.end_token = 151643 # EOS Token ID
        self.meta.dtype = 0 # 假设 FP32=0, 具体需要映射 llaisysDataType_t
        
        # 2. Create Model
        self.model_ptr = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta), 
            device.value, None, 0
        )
        
        # 3. Get Weights Pointer
        self.weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model_ptr)
        self.weights = self.weights_ptr.contents
        
        # Keep python references to tensors to prevent GC
        self.tensors = []

        # 4. Load Weights
        print("Loading weights...")
        for file in sorted(model_path.glob("*.safetensors")):
            with safetensors.numpy.safe_open(file, framework="numpy", device="cpu") as f:
                for name in f.keys():
                    # Load numpy array
                    np_data = f.get_tensor(name)
                    # Convert to LLAISYS Tensor
                    tensor = Tensor.from_numpy(np_data) # 假设有这个helper，或者手动from_list
                    self.tensors.append(tensor)
                    lib_tensor = tensor.lib_tensor()

                    # Map name to C struct
                    # model.embed_tokens.weight
                    if name == "model.embed_tokens.weight":
                        self.weights.in_embed = lib_tensor
                    # lm_head.weight
                    elif name == "lm_head.weight":
                        self.weights.out_embed = lib_tensor
                    # model.norm.weight
                    elif name == "model.norm.weight":
                        self.weights.out_norm_w = lib_tensor
                    # Layers
                    elif name.startswith("model.layers."):
                        parts = name.split(".")
                        idx = int(parts[2]) # layers.0
                        layer_type = parts[3] # self_attn, mlp, input_layernorm
                        
                        if layer_type == "input_layernorm":
                            self.weights.attn_norm_w[idx] = lib_tensor
                        elif layer_type == "post_attention_layernorm":
                            self.weights.mlp_norm_w[idx] = lib_tensor
                        elif layer_type == "self_attn":
                            proj = parts[4] # q_proj, k_proj...
                            if "q_proj" in proj: self.weights.attn_q_w[idx] = lib_tensor
                            elif "k_proj" in proj: self.weights.attn_k_w[idx] = lib_tensor
                            elif "v_proj" in proj: self.weights.attn_v_w[idx] = lib_tensor
                            elif "o_proj" in proj: self.weights.attn_o_w[idx] = lib_tensor
                        elif layer_type == "mlp":
                            proj = parts[4]
                            if "gate_proj" in proj: self.weights.mlp_gate_w[idx] = lib_tensor
                            elif "up_proj" in proj: self.weights.mlp_up_w[idx] = lib_tensor
                            elif "down_proj" in proj: self.weights.mlp_down_w[idx] = lib_tensor

    def __del__(self):
        if hasattr(self, "model_ptr"):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model_ptr)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 100,
        top_k: int = 1, # Not implemented in C API yet (always argmax)
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        current_ids = list(inputs)
        new_tokens = []
        
        # 1. Prefill
        input_array = (ctypes.c_int64 * len(current_ids))(*current_ids)
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self.model_ptr, input_array, len(current_ids)
        )
        
        current_ids.append(next_token)
        new_tokens.append(next_token)
        
        # 2. Decode Loop
        for _ in range(max_new_tokens - 1):
            if next_token == self.meta.end_token:
                break
                
            input_array = (ctypes.c_int64 * 1)(*[next_token])
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.model_ptr, input_array, 1
            )
            
            new_tokens.append(next_token)
            
        return new_tokens
