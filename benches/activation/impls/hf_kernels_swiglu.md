---
on_github: huggingface/kernels-benchmarks
on_huggingface: kernels-community/activation
---

# HF Kernels - SwiGLU Activation

## GPU Info

```python id=nv
import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
```

## SwiGLU Benchmark

```python id=benchmark outputs=activation.jsonl
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

# Load the activation kernel
activation = get_kernel("kernels-community/activation")


def hf_kernels_swiglu(input_tensor):
    hidden_dim = input_tensor.shape[-1] // 2
    out_shape = input_tensor.shape[:-1] + (hidden_dim,)
    out = torch.empty(out_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    return activation.silu_and_mul(out, input_tensor)


run_benchmark(
    kernel_type=KernelTypeEnum.ACTIVATION,
    impl_name="hf_kernels_swiglu",
    impl_tags={"family": "hf-kernels", "backend": "cuda"},
    impl_func=hf_kernels_swiglu,
)
```
