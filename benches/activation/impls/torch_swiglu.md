---
on_github: huggingface/kernels-uvnotes
---

# PyTorch Native - SwiGLU Activation

## GPU Info

```python id=nv
import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
```

## SwiGLU Benchmark (PyTorch Native)

```python id=benchmark outputs=activation.jsonl
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
import torch, torch.nn.functional as F


def swiglu_eager(x):
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


run_benchmark(
    kernel_type=KernelTypeEnum.ACTIVATION,
    impl_name="torch_eager",
    impl_tags={"family":"hf-kernels", "backend":"eager"},
    impl_func=swiglu_eager,
)
```
