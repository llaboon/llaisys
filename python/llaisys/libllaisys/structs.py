# python/llaisys/libllaisys/models.py

import ctypes
from . import LIB_LLAISYS
from .tensor import llaisysTensor_t

# C struct definitions matching include/llaisys/models/qwen2.h

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

# Function Prototypes
LIB_LLAISYS.llaisysQwen2ModelCreate.argtypes = [
    ctypes.POINTER(LlaisysQwen2Meta),
    ctypes.c_int,           # llaisysDeviceType_t
    ctypes.POINTER(ctypes.c_int), # device_ids
    ctypes.c_int            # ndevice
]
LIB_LLAISYS.llaisysQwen2ModelCreate.restype = ctypes.c_void_p

LIB_LLAISYS.llaisysQwen2ModelDestroy.argtypes = [ctypes.c_void_p]
LIB_LLAISYS.llaisysQwen2ModelDestroy.restype = None

LIB_LLAISYS.llaisysQwen2ModelWeights.argtypes = [ctypes.c_void_p]
LIB_LLAISYS.llaisysQwen2ModelWeights.restype = ctypes.POINTER(LlaisysQwen2Weights)

LIB_LLAISYS.llaisysQwen2ModelInfer.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int64), # token_ids
    ctypes.c_size_t                 # ntoken
]
LIB_LLAISYS.llaisysQwen2ModelInfer.restype = ctypes.c_int64
