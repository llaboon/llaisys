from typing import Sequence
import json
import numpy as np
import ctypes
from pathlib import Path
import struct

# 1. 修改导入：只导入确定存在的 DeviceType 和 LIB_LLAISYS
from ..libllaisys import LIB_LLAISYS, DeviceType
from ..libllaisys.models.qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights
from .. import Tensor

# 2. 手动定义数据类型枚举 (对应 C++ enum llaisysDataType_t)
# 警告：这些数字必须与 C++ 头文件中的定义一致！
# 通常约定：0=F32, 1=F16, 2=BF16, 3=I8, 4=I32, 5=I64
# 如果你的 C++ 定义不同，请修改这里
LLAISYS_DTYPE_F32 = 0
LLAISYS_DTYPE_F16 = 1
LLAISYS_DTYPE_BF16 = 2
LLAISYS_DTYPE_I8 = 3
LLAISYS_DTYPE_I32 = 4
LLAISYS_DTYPE_I64 = 5 

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
        # 告诉底层这是 BF16
        dtype = LLAISYS_DTYPE_BF16
    elif np_data.dtype == np.int64:
        dtype = LLAISYS_DTYPE_I64
    elif np_data.dtype == np.int32:
        dtype = LLAISYS_DTYPE_I32
    else:
        # 默认回退到 F32
        print(f"Warning: mapping unknown dtype {np_data.dtype} to F32")
        dtype = LLAISYS_DTYPE_F32

    # 2. 获取形状
    shape = list(np_data.shape)
    
    # 3. 创建空的 LLAISYS Tensor
    # 尝试多种创建方式，因为不知道 Tensor 类的具体实现
    try:
        # 尝试方式 A: 静态工厂方法 (常见于本课程)
        tensor = Tensor.create(shape, dtype, DeviceType.CPU)
    except AttributeError:
        try:
            # 尝试方式 B: 构造函数
            tensor = Tensor(shape, dtype, DeviceType.CPU)
        except TypeError:
            # 尝试方式 C: 也许不需要 device 参数
            tensor = Tensor(shape, dtype)

    # 4. 复制数据
    # 获取 numpy 数据的内存指针
    src_ptr = np_data.ctypes.data_as(ctypes.c_void_p)
    
    # 将数据加载到 Tensor
    # 尝试调用 tensor.load 或使用底层 C API
    if hasattr(tensor, 'load'):
        tensor.load(src_ptr)
    else:
        # 如果 Tensor 类没有 load 方法，尝试直接调用底层 C API
        # 假设 LIB_LLAISYS 暴露了 llaisysTensorCopyFromHost 或类似函数
        # 这里为了保险，我们可以尝试访问内部指针 (如果 Python 包装器暴露了它)
        pass 
        # 注意：如果 Tensor 类没有 load 方法，这里可能需要补充 C API 调用
        # 但通常 wrapper 都会提供 load。
        
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
        self.meta.dtype = LLAISYS_DTYPE_F32 # 默认 FP32
        
        # 2. Create Model
        self.model_ptr = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta), 
            device.value, None, 0
        )
        
        # 3. Get Weights Pointer
        self.weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model_ptr)
        self.weights = self.weights_ptr.contents
        
        # Keep python references
        self.tensors = []

        # 4. Load Weights (手动解析 Safetensors)
        print("Loading weights...")
        safetensor_files = sorted(model_path.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_path}")

        for file in safetensor_files:
            with open(file, "rb") as f_r:
                # 4.1 读取头部长度
                header_len_bytes = f_r.read(8)
                header_len = struct.unpack('<Q', header_len_bytes)[0]
                
                # 4.2 读取 Header JSON
                header_json_bytes = f_r.read(header_len)
                header = json.loads(header_json_bytes)
                
                # 4.3 数据区偏移
                data_section_start = 8 + header_len
                
                for name, meta in header.items():
                    if name == "__metadata__": continue
                    
                    dtype_str = meta['dtype']
                    shape = meta['shape']
                    start_offset, end_offset = meta['data_offsets']
                    
                    # 4.4 定位并读取
                    f_r.seek(data_section_start + start_offset)
                    length = end_offset - start_offset
                    raw_bytes = f_r.read(length)
                    
                    # 4.5 Numpy 转换 (核心：bf16 -> uint16)
                    if dtype_str == "BF16" or dtype_str == "bfloat16":
                        np_data = np.frombuffer(raw_bytes, dtype=np.uint16)
                    elif dtype_str == "F32" or dtype_str == "float32":
                        np_data = np.frombuffer(raw_bytes, dtype=np.float32)
                    elif dtype_str == "F16" or dtype_str == "float16":
                        np_data = np.frombuffer(raw_bytes, dtype=np.float16)
                    elif dtype_str == "I64" or dtype_str == "int64":
                         np_data = np.frombuffer(raw_bytes, dtype=np.int64)
                    else:
                        print(f"Warning: Unknown dtype {dtype_str}, treating as F32")
                        np_data = np.frombuffer(raw_bytes, dtype=np.float32)
                    
                    np_data = np_data.reshape(shape)

                    # 4.6 转换为 LLAISYS Tensor
                    # 确保内存连续，否则传指针会错
                    if not np_data.flags['C_CONTIGUOUS']:
                        np_data = np.ascontiguousarray(np_data)

                    tensor = numpy_to_tensor(np_data)
                    self.tensors.append(tensor)
                    lib_tensor = tensor.lib_tensor()

                    # 4.7 映射到 C 结构体
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
