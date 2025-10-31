---
on_github: huggingface/kernels-benchmarks
---

# PyTorch Native - Causal Conv1D

## GPU Info

```python id=nv
import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
```

## Causal Conv1D Benchmark (PyTorch Native)

```python id=benchmark outputs=causal_conv1d.jsonl
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
import torch.nn.functional as F
import sys
from kernels_benchmark_tools import KernelTypeEnum, run_benchmark


def torch_causal_conv1d(input_tensor, weight, bias):
    # Convert to weight dtype for computation
    x = input_tensor.to(weight.dtype)
    dim = weight.shape[0]
    width = weight.shape[1]
    seqlen = input_tensor.shape[-1]

    # Depthwise causal conv1d using PyTorch
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)

    # Truncate to original sequence length
    out = out[..., :seqlen]

    # Convert back to original dtype
    return out.to(input_tensor.dtype)


run_benchmark(
    kernel_type=KernelTypeEnum.CAUSAL_CONV1D,
    impl_name="torch_eager",
    impl_tags={"family": "pytorch", "backend": "eager"},
    impl_func=torch_causal_conv1d,
)
```
