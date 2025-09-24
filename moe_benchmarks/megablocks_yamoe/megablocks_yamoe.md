---
title: "uvnote Integration Test Report"
author: "uvnote"
theme: "light"
syntax_theme: "monokai"
show_line_numbers: true
collapse_code: false
custom_css: |
    #output-setup {
        overflow-x: auto;
    }
    .cell-output {
        overflow: scroll;
    }
    .cell-stdout {
        width: max-content;
        overflow: scroll;
    }
    .cell-stderr {
        width: max-content;
        overflow: scroll;
        max-height: 300px;
    }
---

```python id=nv
import subprocess

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
```

# Comparison of Megablocks and Yamoe Kernels

This note compares the performance of the Megablocks and Yamoe kernels on the GPT-OSS-20B model.

## Megablocks kernel

```python id=setup2
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "accelerate>=1.10.1",
#     "torch>=2.7.0",
#     "kernels==0.10.0",
#     "transformers@https://github.com/huggingface/transformers.git",
#     "ipdb>=0.13.13",
#     "matplotlib>=3.7.2",
#     "numpy>=1.24.3",
# ]
# ///

import torch
from transformers import GptOssForCausalLM, PreTrainedTokenizerFast, Mxfp4Config
import time
import torch.nn as nn
from kernels import register_kernel_mapping, Mode, LayerRepository
import sys
import torch.profiler
import gc
import logging

# set to debug logging
logging.basicConfig(level=logging.INFO)

def reset_peak_memory_stats():
    """Clear CUDA cache and reset memory allocation counters."""
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

def get_memory_stats():
    """Get current and peak CUDA memory usage."""
    if not torch.cuda.is_available():
        return {"allocated_gb": 0, "peak_gb": 0, "reserved_gb": 0}
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
    }

def override_kernel_layer_name(cls_name: str, value) -> bool:
    """Helper to dynamically override the kernel_layer_name in a model class."""
    for mod in sys.modules.values():
        if mod is None:
            continue
        obj = getattr(mod, cls_name, None)
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            setattr(obj, "kernel_layer_name", value)
            print(f"Overrode {cls_name}.kernel_layer_name to {value}")
            return True
    return False


# Init the model the normal way
model_id = "openai/gpt-oss-20b"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
quantization_config = Mxfp4Config(dequantize=True)


from kernels import replace_kernel_forward_from_hub, register_kernel_mapping, LayerRepository, Mode

from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP, GptOssRMSNorm

replace_kernel_forward_from_hub(GptOssRMSNorm, None)  # direct, type-safe
custom_mapping = {
    "Yamoe": {
        "cuda": {
            Mode.INFERENCE: LayerRepository(
                repo_id="drbh/yamoe",
                layer_name="Yamoe",
                revision="v0.3.0",
            )
        }
    }
}
register_kernel_mapping(custom_mapping)


model = GptOssForCausalLM.from_pretrained(
    model_id,
    dtype="bfloat16",
    device_map="auto",
    use_kernels=True,
    quantization_config=quantization_config,
).eval()

messages = [
    {"role": "system", "content": "What is Tensor Parallelism?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="low",
).to("cuda")

max_tokens = 256

with torch.inference_mode():
    start_time = time.perf_counter()
    generated = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=None,
    )
    end_time = time.perf_counter()

print(tokenizer.decode(generated[0], skip_special_tokens=False))
print(f"Generation took {end_time - start_time:.2f} seconds")

```

## Yamoe Kernel

```python id=setup
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "accelerate>=1.10.1",
#     "torch>=2.7.0",
#     "kernels==0.10.0",
#     "transformers@https://github.com/huggingface/transformers.git",
#     "ipdb>=0.13.13",
#     "matplotlib>=3.7.2",
#     "numpy>=1.24.3",
# ]
# ///

import torch
from transformers import GptOssForCausalLM, PreTrainedTokenizerFast, Mxfp4Config
import time
import torch.nn as nn
from kernels import register_kernel_mapping, Mode, LayerRepository
import sys
import torch.profiler
import gc
import logging

# set to debug logging
logging.basicConfig(level=logging.INFO)

def reset_peak_memory_stats():
    """Clear CUDA cache and reset memory allocation counters."""
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

def get_memory_stats():
    """Get current and peak CUDA memory usage."""
    if not torch.cuda.is_available():
        return {"allocated_gb": 0, "peak_gb": 0, "reserved_gb": 0}
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
    }

def override_kernel_layer_name(cls_name: str, value) -> bool:
    """Helper to dynamically override the kernel_layer_name in a model class."""
    for mod in sys.modules.values():
        if mod is None:
            continue
        obj = getattr(mod, cls_name, None)
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            setattr(obj, "kernel_layer_name", value)
            print(f"Overrode {cls_name}.kernel_layer_name to {value}")
            return True
    return False


# Init the model the normal way
model_id = "openai/gpt-oss-20b"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
quantization_config = Mxfp4Config(dequantize=True)


from kernels import replace_kernel_forward_from_hub, register_kernel_mapping, LayerRepository, Mode

from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP, GptOssRMSNorm

replace_kernel_forward_from_hub(GptOssMLP, "Yamoe")
replace_kernel_forward_from_hub(GptOssRMSNorm, None)
custom_mapping = {
    "Yamoe": {
        "cuda": {
            Mode.INFERENCE: LayerRepository(
                repo_id="drbh/yamoe",
                layer_name="Yamoe",
                revision="v0.3.0",
            )
        }
    }
}
register_kernel_mapping(custom_mapping)


model = GptOssForCausalLM.from_pretrained(
    model_id,
    dtype="bfloat16",
    device_map="auto",
    use_kernels=True,
    quantization_config=quantization_config,
).eval()

messages = [
    {"role": "system", "content": "What is Tensor Parallelism?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="low",
).to("cuda")

max_tokens = 256

with torch.inference_mode():
    start_time = time.perf_counter()
    generated = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=None,
    )
    end_time = time.perf_counter()

print(tokenizer.decode(generated[0], skip_special_tokens=False))
print(f"Generation took {end_time - start_time:.2f} seconds")

```
