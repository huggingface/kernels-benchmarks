---
on_github: huggingface/kernels-benchmarks
---

# Torch LayerNorm Implementation

## GPU Info

```python id=nv
import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
```

## LayerNorm Benchmark (PyTorch)

```python id=benchmark outputs=layer_norm.jsonl
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


def torch_layer_norm(x, weight, bias, eps: float = 1e-5):
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)


run_benchmark(
    kernel_type=KernelTypeEnum.LAYER_NORM,
    impl_name="torch_layer_norm",
    impl_tags={"family": "torch", "op": "layer_norm"},
    impl_func=torch_layer_norm,
)
```
