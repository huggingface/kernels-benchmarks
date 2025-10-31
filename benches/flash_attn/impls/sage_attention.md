---
on_github: huggingface/kernels-benchmarks
on_huggingface: kernels-community/sage_attention
---

# SageAttention Implementation

## SageAttention Benchmark (INT8 Quantized)

```python id=benchmark outputs=attention.jsonl
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "torch==2.8.0",
#     "kernels",
#     "kernels-benchmark-tools",
# ]
#
# [tool.uv.sources]
# kernels-benchmark-tools = { path = "../../../../../tools", editable = true }
# ///
import torch
import sys
from kernels_benchmark_tools import KernelTypeEnum, run_benchmark
from kernels import get_kernel

# Load the sage attention kernel
hf_kernels_sage_attn = get_kernel("kernels-community/sage_attention")


def sage_attention(query, key, value):
    """SageAttention with INT8 Q/K quantization and FP16 P/V"""
    return hf_kernels_sage_attn.fwd(query, key, value, is_causal=False)[0]


run_benchmark(
    kernel_type=KernelTypeEnum.ATTENTION,
    impl_name="sage_int8_fp16",
    impl_tags={"family": "sageattention", "backend": "int8_fp16_cuda", "compile": "none"},
    impl_func=sage_attention,
)
```
