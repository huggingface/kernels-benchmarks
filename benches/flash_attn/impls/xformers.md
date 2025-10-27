---
on_github: huggingface/kernels-uvnotes
---

# xFormers Memory Efficient Attention

## xFormers Benchmark

```python id=benchmark outputs=attention.jsonl
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "torch==2.8.0",
#     "kernels-benchmark-tools",
#     "xformers",
# ]
#
# [tool.uv.sources]
# kernels-benchmark-tools = { path = "../../../../../tools", editable = true }
# ///
import torch
import sys
from kernels_benchmark_tools import KernelTypeEnum, run_benchmark
import xformers.ops as xops


def xformers_attention(q, k, v):
    """xFormers memory efficient attention"""
    # xFormers expects [batch, seq_len, heads, head_dim]
    return xops.memory_efficient_attention(q, k, v)


run_benchmark(
    kernel_type=KernelTypeEnum.ATTENTION,
    impl_name="xformers_meff",
    impl_tags={"family": "xformers", "backend": "memory_efficient", "compile": "none"},
    impl_func=xformers_attention,
)
```
