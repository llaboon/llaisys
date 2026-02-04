import ctypes
import numpy as np
import os

# =========================================================================
# 【关键修复】: 在导入 safetensors 或加载模型前，强行注册 bfloat16
# 这会欺骗环境，将 bf16 数据视为 uint16 读取，绕过 numpy 版本不支持的问题
# =========================================================================
try:
    if 'bfloat16' not in np.typeDict:
        # 定义 bfloat16 为 2字节无符号整数 (内存布局一致)
        np.typeDict['bfloat16'] = np.dtype('<u2') 
        np.bfloat16 = np.dtype('<u2')
except Exception as e:
    print(f"[Warning] Failed to patch numpy for bfloat16: {e}")

# 导入框架库
from .. import llaisysTensor_t
from .. import LIB_LLAISYS

# 对应 C 里的 LlaisysQwen2Meta
class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", ctypes.c_int), # llaisysDataType_t
        ("nlayer", ctypes.c_size_t),
        ("hs", ctypes.c_size_t),
        ("nh", ctypes.c_size_t),
        ("nkvh", ctypes.c_size_t),
        ("dh", ctypes.c_size_t),
        ("di", ctypes.c_size_t),
        ("maxseq", ctypes.c_size_t),
        ("voc", ctypes.c_size_t),
        ("epsilon", ctypes.c_float),
        ("theta", ctypes.c_float),
        ("end_token", ctypes.c_int64),
    ]

# 对应 C 里的 LlaisysQwen2Weights
class LlaisysQwen2Weights(ctypes.Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_q_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_q_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_k_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_k_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_v_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_v_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_o_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_norm_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_gate_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_up_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_down_w", ctypes.POINTER(llaisysTensor_t)),
    ]

# 绑定 C API 函数
LIB_LLAISYS.llaisysQwen2ModelCreate.argtypes = [ctypes.POINTER(LlaisysQwen2Meta), ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
LIB_LLAISYS.llaisysQwen2ModelCreate.restype = ctypes.c_void_p

LIB_LLAISYS.llaisysQwen2ModelDestroy.argtypes = [ctypes.c_void_p]
LIB_LLAISYS.llaisysQwen2ModelDestroy.restype = None

LIB_LLAISYS.llaisysQwen2ModelWeights.argtypes = [ctypes.c_void_p]
LIB_LLAISYS.llaisysQwen2ModelWeights.restype = ctypes.POINTER(LlaisysQwen2Weights)

LIB_LLAISYS.llaisysQwen2ModelInfer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t]
LIB_LLAISYS.llaisysQwen2ModelInfer.restype = ctypes.c_int64

# 定义 Python 端的 Model 类 (供 test_infer.py 调用)
class Qwen2:
    def __init__(self, model_path, device_info):
        # 这里需要解析 config.json 并填充 meta，这部分作业通常提供了基础框架
        # 假设上层调用者已经处理了 meta 的构建
        pass 
        # 注意：由于作业逻辑是在 test_infer.py 中调用 load_llaisys_model
        # 实际的 Python 包装逻辑通常在这里实现，将 safetensors 数据传给 C++
        # 但因为作业要求很具体，核心在于上面的 numpy patch 和下面的 C++ 实现
