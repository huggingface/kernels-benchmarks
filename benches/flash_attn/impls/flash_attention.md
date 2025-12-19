---
on_github: huggingface/kernels-benchmarks
platforms:
  - linux
---

# Flash Attention Implementation

## GPU Info

```python id=nv
import subprocess

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
```

## Flash Attention Benchmark

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


def torch_flash(q, k, v):
    qt, kt, vt = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        o = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt)
    return o.transpose(1, 2).contiguous()


run_benchmark(
    kernel_type=KernelTypeEnum.ATTENTION,
    impl_name="torch_flash_ma",
    impl_tags={"family": "torch-sdpa", "backend": "FLASH", "compile": "max-autotune"},
    impl_func=torch_flash,
)
```
