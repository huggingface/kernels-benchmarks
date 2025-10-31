---
on_github: huggingface/kernels-benchmarks
on_huggingface: kernels-community/causal-conv1d
---

# HF Kernels - Causal Conv1D

## GPU Info

```python id=nv
import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
```

## Causal Conv1D Benchmark

```python id=benchmark outputs=causal_conv1d.jsonl
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

# Load the causal conv1d kernel
causal_conv1d = get_kernel("kernels-community/causal-conv1d")


def hf_kernels_causal_conv1d(input_tensor, weight, bias):
    return causal_conv1d.causal_conv1d_fn(input_tensor, weight, bias)


run_benchmark(
    kernel_type=KernelTypeEnum.CAUSAL_CONV1D,
    impl_name="hf_kernels_causal_conv1d",
    impl_tags={"family": "hf-kernels", "backend": "cuda"},
    impl_func=hf_kernels_causal_conv1d,
)
```
