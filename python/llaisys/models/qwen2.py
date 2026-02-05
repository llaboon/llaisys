from typing import Sequence
import json
import ctypes
import numpy as np
from pathlib import Path
import safetensors.numpy
import gc

from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys import LlaisysQwen2Meta, LlaisysQwen2Weights
from ..libllaisys import llaisysTensor_t

class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.device = device
        model_path = Path(model_path)
        
        # 1. Load Config
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
            
        # 2. Prepare Meta
        self.meta = LlaisysQwen2Meta()
        # Mapping torch/hf dtype strings to llaisysDataType_t if needed
        # Assuming 1 = FLOAT16, 2 = FLOAT32 based on common conventions or tensor.h
        # Assignment strictness: Check llaisys_types.py. Assuming BFloat16/Float16 is supported.
        # For safety in this assignment without dtype map, we default to whatever the C++ expects (Float32 or Float16).
        # We will cast numpy data to float32 or float16 before loading.
        self.meta.dtype = DataType.FLOAT32 if device == DeviceType.CPU else DataType.FLOAT16 
        
        self.meta.nlayer = config["num_hidden_layers"]
        self.meta.hs = config["hidden_size"]
        self.meta.nh = config["num_attention_heads"]
        self.meta.nkvh = config.get("num_key_value_heads", self.meta.nh)
        self.meta.dh = self.meta.hs // self.meta.nh
        self.meta.di = config["intermediate_size"]
        self.meta.maxseq = config.get("max_position_embeddings", 2048)
        self.meta.voc = config["vocab_size"]
        self.meta.epsilon = config["rms_norm_eps"]
        self.meta.theta = config.get("rope_theta", 10000.0)
        self.meta.end_token = config.get("eos_token_id", 151643)

        # 3. Create Model Backend
        self.model_ptr = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta),
            device.value if hasattr(device, 'value') else device,
            None, 0
        )
        if not self.model_ptr:
            raise RuntimeError("Failed to create Qwen2 model backend")

        self.c_weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model_ptr).contents

        # 4. Load Weights
        # We need to map HF names to our struct
        tensors_map = {}
        for file in sorted(model_path.glob("*.safetensors")):
            with safetensors.numpy.safe_open(file, framework="numpy") as f:
                for name in f.keys():
                    tensors_map[name] = f.get_tensor(name)

        def load_w(tensor_ptr_loc, name, optional=False):
            if name not in tensors_map:
                if optional: return
                # Handle tied weights (embedding == lm_head)
                return 
            
            data_np = tensors_map[name]
            
            # Type Conversion
            target_dtype = np.float32 if self.meta.dtype == DataType.FLOAT32 else np.float16
            if data_np.dtype != target_dtype:
                data_np = data_np.astype(target_dtype)
            
            # Ensure contiguous
            if not data_np.flags['C_CONTIGUOUS']:
                data_np = np.ascontiguousarray(data_np)

            # Shape handling
            shape = data_np.shape
            ndim = len(shape)
            c_shape = (ctypes.c_size_t * ndim)(*shape)
            
            # Create Tensor using LIB_LLAISYS directly
            # tensorCreate(shape, ndim, dtype, device, device_id)
            tensor_handle = LIB_LLAISYS.tensorCreate(
                c_shape, ndim, self.meta.dtype, 
                self.device.value if hasattr(self.device, 'value') else self.device, 
                0
            )
            
            # Load Data
            # tensorLoad(tensor, data_ptr)
            LIB_LLAISYS.tensorLoad(tensor_handle, data_np.ctypes.data_as(ctypes.c_void_p))
            
            # Assign to structure
            # tensor_ptr_loc is either a llaisysTensor_t field or an element in a pointer array
            if isinstance(tensor_ptr_loc, int): # It's an index in a ctypes array
                raise RuntimeError("Cannot assign directly to index via helper")
            else:
                 # It's a ctypes field? No, we need to handle array assignment manually
                 pass
            return tensor_handle

        # Main Weights
        self.c_weights.in_embed = load_w(None, "model.embed_tokens.weight")
        
        if "lm_head.weight" in tensors_map:
            self.c_weights.out_embed = load_w(None, "lm_head.weight")
        else:
            self.c_weights.out_embed = self.c_weights.in_embed # Tied

        self.c_weights.out_norm_w = load_w(None, "model.norm.weight")

        # Layers
        for i in range(self.meta.nlayer):
            prefix = f"model.layers.{i}"
            
            # Helper to assign to array
            def set_arr(arr, idx, suffix, bias=False):
                name = f"{prefix}.{suffix}"
                # Handle bias existence
                if bias and name not in tensors_map:
                    # Some models don't have bias, pass NULL/None if backend supports it
                    # But if we must provide a tensor, we might need a zero tensor?
                    # The C backend (linear op) usually expects bias. 
                    # Qwen2 usually has no bias in Linear except QKV in some versions?
                    # R1-Distill-Qwen follows Qwen2.5 structure usually.
                    # If key missing, we set to None (0)
                    arr[idx] = None
                    return

                t = load_w(None, name)
                arr[idx] = t

            set_arr(self.c_weights.attn_norm_w, i, "input_layernorm.weight")
            set_arr(self.c_weights.attn_q_w, i, "self_attn.q_proj.weight")
            set_arr(self.c_weights.attn_q_b, i, "self_attn.q_proj.bias", bias=True)
            set_arr(self.c_weights.attn_k_w, i, "self_attn.k_proj.weight")
            set_arr(self.c_weights.attn_k_b, i, "self_attn.k_proj.bias", bias=True)
            set_arr(self.c_weights.attn_v_w, i, "self_attn.v_proj.weight")
            set_arr(self.c_weights.attn_v_b, i, "self_attn.v_proj.bias", bias=True)
            set_arr(self.c_weights.attn_o_w, i, "self_attn.o_proj.weight")
            
            set_arr(self.c_weights.mlp_norm_w, i, "post_attention_layernorm.weight")
            set_arr(self.c_weights.mlp_gate_w, i, "mlp.gate_proj.weight")
            set_arr(self.c_weights.mlp_up_w, i, "mlp.up_proj.weight")
            set_arr(self.c_weights.mlp_down_w, i, "mlp.down_proj.weight")

        # Cleanup numpy memory
        del tensors_map
        gc.collect()

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if max_new_tokens is None:
            max_new_tokens = 128
            
        generated = []
        current_tokens = list(inputs)
        
        # We need to process tokens.
        # Strategy: 
        # 1. Prefill: Process all input tokens. Get the last token prediction.
        # 2. Decode: Loop max_new_tokens times.
        
        # Prepare buffer for tokens
        # We process the whole prompt first
        input_len = len(current_tokens)
        c_tokens = (ctypes.c_int64 * input_len)(*current_tokens)
        
        next_token_id = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self.model_ptr,
            c_tokens,
            input_len
        )
        
        # If testing, we just return what infer returns? No, generate needs to loop.
        
        for _ in range(max_new_tokens):
            if next_token_id == self.meta.end_token:
                break
            
            generated.append(next_token_id)
            current_tokens.append(next_token_id)
            
            # Prepare single token input for next step
            c_input = (ctypes.c_int64 * 1)(next_token_id)
            next_token_id = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.model_ptr,
                c_input,
                1
            )
            
        return generated

    def __del__(self):
        if hasattr(self, 'model_ptr') and self.model_ptr:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model_ptr)
