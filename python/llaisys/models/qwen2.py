from typing import Sequence
import json
import numpy as np
import ctypes
from pathlib import Path
import struct

# 注意：我们这里不需要 import safetensors.numpy 了，因为我们将手动解析
from ..libllaisys import LIB_LLAISYS, DeviceType
from ..libllaisys.models.qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights
from .. import Tensor

class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        
        # 1. Load Config
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")

        with open(config_path, "r") as f:
            cfg = json.load(f)
            
        self.meta = LlaisysQwen2Meta()
        self.meta.nlayer = cfg["num_hidden_layers"]
        self.meta.hs = cfg["hidden_size"]
        self.meta.nh = cfg["num_attention_heads"]
        self.meta.nkvh = cfg["num_key_value_heads"]
        self.meta.dh = self.meta.hs // self.meta.nh
        self.meta.di = cfg["intermediate_size"]
        self.meta.maxseq = 2048
        self.meta.voc = cfg["vocab_size"]
        self.meta.epsilon = cfg["rms_norm_eps"]
        self.meta.theta = cfg.get("rope_theta", 10000.0)
        self.meta.end_token = 151643 
        self.meta.dtype = 0 # 假设 0=FP32 (虽然数据是BF16，但在C++里会转)
        
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

        # 4. Load Weights (手动解析 Safetensors，绕过版本兼容性问题)
        print("Loading weights...")
        safetensor_files = sorted(model_path.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_path}")

        for file in safetensor_files:
            with open(file, "rb") as f_r:
                # 4.1 读取头部长度 (8字节, uint64, 小端序)
                header_len_bytes = f_r.read(8)
                header_len = struct.unpack('<Q', header_len_bytes)[0]
                
                # 4.2 读取 Header JSON
                header_json_bytes = f_r.read(header_len)
                header = json.loads(header_json_bytes)
                
                # 4.3 计算数据区的绝对起始偏移量
                data_section_start = 8 + header_len
                
                # 4.4 遍历 Header 中的每个张量
                for name, meta in header.items():
                    # 跳过元数据键
                    if name == "__metadata__": continue
                    
                    dtype_str = meta['dtype']
                    shape = meta['shape']
                    start_offset, end_offset = meta['data_offsets']
                    
                    # 4.5 定位到该张量的数据位置
                    f_r.seek(data_section_start + start_offset)
                    
                    # 4.6 读取原始字节
                    length = end_offset - start_offset
                    raw_bytes = f_r.read(length)
                    
                    # 4.7 转换为 Numpy 数组 (核心解法)
                    # 无论 Numpy 版本多旧，它都支持 uint16
                    if dtype_str == "BF16" or dtype_str == "bfloat16":
                        np_data = np.frombuffer(raw_bytes, dtype=np.uint16)
                    elif dtype_str == "F32" or dtype_str == "float32":
                        np_data = np.frombuffer(raw_bytes, dtype=np.float32)
                    elif dtype_str == "F16" or dtype_str == "float16":
                        np_data = np.frombuffer(raw_bytes, dtype=np.float16)
                    elif dtype_str == "I64" or dtype_str == "int64":
                         np_data = np.frombuffer(raw_bytes, dtype=np.int64)
                    else:
                        print(f"Warning: Unknown dtype {dtype_str} for {name}, treating as float32")
                        np_data = np.frombuffer(raw_bytes, dtype=np.float32)
                    
                    # 恢复形状
                    np_data = np_data.reshape(shape)

                    # --- 后续逻辑保持不变 ---
                    
                    # Convert to LLAISYS Tensor
                    tensor = Tensor.from_numpy(np_data) 
                    self.tensors.append(tensor)
                    lib_tensor = tensor.lib_tensor()

                    # Map name to C struct
                    if name == "model.embed_tokens.weight":
                        self.weights.in_embed = lib_tensor
                    elif name == "lm_head.weight":
                        self.weights.out_embed = lib_tensor
                    elif name == "model.norm.weight":
                        self.weights.out_norm_w = lib_tensor
                    elif name.startswith("model.layers."):
                        parts = name.split(".")
                        idx = int(parts[2]) 
                        layer_type = parts[3] 
                        
                        # 确保只加载 weight，不处理 bias (Qwen2 通常无 bias，防止同名覆盖)
                        if name.endswith(".weight"):
                            if layer_type == "input_layernorm":
                                self.weights.attn_norm_w[idx] = lib_tensor
                            elif layer_type == "post_attention_layernorm":
                                self.weights.mlp_norm_w[idx] = lib_tensor
                            elif layer_type == "self_attn":
                                proj = parts[4] 
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
        if hasattr(self, "model_ptr") and self.model_ptr:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model_ptr)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 100,
        top_k: int = 1, 
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
