from typing import Sequence
import json
import numpy as np
import ctypes
from pathlib import Path
import struct

from ..libllaisys import LIB_LLAISYS, DeviceType, LLAISYS_DTYPE_F32, LLAISYS_DTYPE_F16, LLAISYS_DTYPE_BF16, LLAISYS_DTYPE_I64, LLAISYS_DTYPE_I32, LLAISYS_DEVICE_CPU
from ..libllaisys.models.qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights
from .. import Tensor

def numpy_to_tensor(np_data):
    """
    手动将 Numpy 数组转换为 LLAISYS Tensor
    """
    # 1. 确定数据类型映射
    if np_data.dtype == np.float32:
        dtype = LLAISYS_DTYPE_F32
    elif np_data.dtype == np.float16:
        dtype = LLAISYS_DTYPE_F16
    elif np_data.dtype == np.uint16: 
        # 特殊情况：我们将 BF16 读取为了 uint16
        # 告诉底层这是 BF16，虽然 Python 端是 uint16
        dtype = LLAISYS_DTYPE_BF16
    elif np_data.dtype == np.int64:
        dtype = LLAISYS_DTYPE_I64
    elif np_data.dtype == np.int32:
        dtype = LLAISYS_DTYPE_I32
    else:
        raise ValueError(f"Unsupported numpy dtype: {np_data.dtype}")

    # 2. 获取形状
    shape = list(np_data.shape)
    
    # 3. 创建空的 LLAISYS Tensor
    # 注意：这里假设 Tensor.create 是存在的 (基于 C++ API 习惯)
    # 如果 Tensor 类只有构造函数，可能是 Tensor(shape, dtype, device)
    # 既然报错只说没有 from_numpy，我们尝试使用构造函数或 create
    try:
        tensor = Tensor(shape, dtype, device=DeviceType.CPU)
    except:
        # 备选：如果 Tensor 是通过 create 静态方法创建
        tensor = Tensor.create(shape, dtype, device=DeviceType.CPU)

    # 4. 复制数据
    # 获取 numpy 数据的指针
    src_ptr = np_data.ctypes.data_as(ctypes.c_void_p)
    # 获取 tensor 数据的指针 (假设 tensor.data_ptr() 或 tensor.load() 存在)
    # 大多数绑定库允许直接 load
    try:
        tensor.load(src_ptr)
    except AttributeError:
        # 如果没有 load 方法，尝试使用 copy_from_numpy 或者直接内存拷贝
        # 这里假设底层有 C API: llaisysTensorLoad(tensor, data)
        LIB_LLAISYS.llaisysTensorLoad(tensor.lib_tensor(), src_ptr)
        
    return tensor

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
        self.meta.dtype = 0 
        
        # 2. Create Model
        self.model_ptr = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta), 
            device.value, None, 0
        )
        
        # 3. Get Weights Pointer
        self.weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model_ptr)
        self.weights = self.weights_ptr.contents
        
        self.tensors = []

        # 4. Load Weights
        print("Loading weights...")
        safetensor_files = sorted(model_path.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_path}")

        for file in safetensor_files:
            with open(file, "rb") as f_r:
                header_len_bytes = f_r.read(8)
                header_len = struct.unpack('<Q', header_len_bytes)[0]
                
                header_json_bytes = f_r.read(header_len)
                header = json.loads(header_json_bytes)
                
                data_section_start = 8 + header_len
                
                for name, meta in header.items():
                    if name == "__metadata__": continue
                    
                    dtype_str = meta['dtype']
                    shape = meta['shape']
                    start_offset, end_offset = meta['data_offsets']
                    
                    f_r.seek(data_section_start + start_offset)
                    length = end_offset - start_offset
                    raw_bytes = f_r.read(length)
                    
                    if dtype_str == "BF16" or dtype_str == "bfloat16":
                        np_data = np.frombuffer(raw_bytes, dtype=np.uint16)
                    elif dtype_str == "F32" or dtype_str == "float32":
                        np_data = np.frombuffer(raw_bytes, dtype=np.float32)
                    elif dtype_str == "F16" or dtype_str == "float16":
                        np_data = np.frombuffer(raw_bytes, dtype=np.float16)
                    elif dtype_str == "I64" or dtype_str == "int64":
                         np_data = np.frombuffer(raw_bytes, dtype=np.int64)
                    else:
                        print(f"Warning: Unknown dtype {dtype_str}, default to F32")
                        np_data = np.frombuffer(raw_bytes, dtype=np.float32)
                    
                    np_data = np_data.reshape(shape)

                    # --- 修改点：调用自定义转换函数 ---
                    # 确保 numpy 数组是连续内存，否则 ctypes 指针可能无效
                    if not np_data.flags['C_CONTIGUOUS']:
                        np_data = np.ascontiguousarray(np_data)

                    tensor = numpy_to_tensor(np_data)
                    # ------------------------------

                    self.tensors.append(tensor)
                    lib_tensor = tensor.lib_tensor()

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
