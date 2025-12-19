---
on_github: huggingface/kernels-benchmarks
on_huggingface: kernels-community/flash-attn3
platforms:
  - linux
---

# HF Kernels - Flash Attention 3

## HuggingFace Kernels Flash Attention 3 Benchmark

```python id=benchmark outputs=attention.jsonl
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "torch==2.8.0",
#     "kernels-benchmark-tools",
#     "kernels",
# ]
#
# [tool.uv.sources]
# kernels-benchmark-tools = { path = "../../../../../tools", editable = true }
# ///
import torch
import sys
from kernels_benchmark_tools import KernelTypeEnum, run_benchmark
from kernels import get_kernel

# Load the flash attention 3 kernel
hf_kernels_flash_attn3 = get_kernel("kernels-community/flash-attn3")


def hf_flash_attention3(query, key, value):
    return hf_kernels_flash_attn3.flash_attn_func(query, key, value, causal=False)[0]


run_benchmark(
    kernel_type=KernelTypeEnum.ATTENTION,
    impl_name="hf_kernels_flash_attn3",
    impl_tags={"family": "hf-kernels", "backend": "flash-attn3", "compile": "none"},
    impl_func=hf_flash_attention3,
)
```
