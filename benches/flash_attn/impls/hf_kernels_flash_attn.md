---
on_github: huggingface/kernels-benchmarks
on_huggingface: kernels-community/flash-attn
---

# HF Kernels - Flash Attention

## HuggingFace Kernels Flash Attention Benchmark

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

# Load the flash attention kernel
hf_kernels_flash_attn = get_kernel("kernels-community/flash-attn")


def hf_flash_attention(query, key, value):
    """HuggingFace Kernels Flash Attention"""
    return hf_kernels_flash_attn.fwd(query, key, value, is_causal=False)[0]


run_benchmark(
    kernel_type=KernelTypeEnum.ATTENTION,
    impl_name="hf_kernels_flash_attn",
    impl_tags={"family": "hf-kernels", "backend": "flash-attn", "compile": "none"},
    impl_func=hf_flash_attention,
)
```
