---
on_github: huggingface/kernels-benchmarks
platforms:
  - linux
---

# Memory Efficient Attention Implementation

## Memory Efficient SDPA Benchmark

```python id=benchmark outputs=attention.jsonl
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "torch==2.8.0",
#     "kernels-benchmark-tools",
# ]
#
# [tool.uv.sources]
# kernels-benchmark-tools = { path = "../../../../../tools", editable = true }
# ///
import torch
import sys
from kernels_benchmark_tools import KernelTypeEnum, run_benchmark


def torch_mem_eff(q, k, v):
    qt, kt, vt = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    with torch.nn.attention.sdpa_kernel(
        torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
    ):
        o = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt)
    return o.transpose(1, 2).contiguous()


run_benchmark(
    kernel_type=KernelTypeEnum.ATTENTION,
    impl_name="torch_mem_eff",
    impl_tags={"family": "torch-sdpa", "backend": "EFFICIENT", "compile": "none"},
    impl_func=torch_mem_eff,
)
```
