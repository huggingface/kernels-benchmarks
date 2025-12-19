---
on_github: huggingface/kernels-benchmarks
platforms:
  - darwin
---

# PyTorch Native - SwiGLU Activation (macOS)

## System Info

```python id=sysinfo
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.8.0",
# ]
# ///
import platform
import subprocess
print(f"Platform: {platform.system()} {platform.machine()}")
print(f"Python: {platform.python_version()}")
# Check for MPS availability
import torch
print(f"PyTorch: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
```

## SwiGLU Benchmark (PyTorch Native - macOS)

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
    impl_name="torch_eager_darwin",
    impl_tags={"family":"pytorch", "backend":"eager", "platform": "darwin"},
    impl_func=swiglu_eager,
)
```
