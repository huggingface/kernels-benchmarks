---
on_github: huggingface/kernels-benchmarks
---

# PyTorch Native - Rotary Position Embeddings

## GPU Info

```python id=nv
import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
```

## Rotary Embeddings Benchmark (PyTorch Native)

```python id=benchmark outputs=rotary.jsonl
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


def apply_rotary_torch(x1, x2, cos, sin, conj=False):
    """Reference rotary implementation."""
    if not conj:
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
    else:
        out1 = x1 * cos + x2 * sin
        out2 = -x1 * sin + x2 * cos
    return out1, out2


def torch_rotary(query, key, cos, sin, conj=False):
    rotary_dim = cos.shape[-1]

    # Clone inputs to avoid modifying them
    q_out = query.clone()
    k_out = key.clone()

    # Apply rotation to query
    q1 = q_out[..., :rotary_dim]
    q2 = q_out[..., rotary_dim : 2 * rotary_dim]
    q_out_1, q_out_2 = apply_rotary_torch(q1, q2, cos, sin, conj)
    q_out[..., :rotary_dim] = q_out_1
    q_out[..., rotary_dim : 2 * rotary_dim] = q_out_2

    # Apply rotation to key
    k1 = k_out[..., :rotary_dim]
    k2 = k_out[..., rotary_dim : 2 * rotary_dim]
    k_out_1, k_out_2 = apply_rotary_torch(k1, k2, cos, sin, conj)
    k_out[..., :rotary_dim] = k_out_1
    k_out[..., rotary_dim : 2 * rotary_dim] = k_out_2

    return q_out, k_out


run_benchmark(
    kernel_type=KernelTypeEnum.ROTARY,
    impl_name="torch_eager",
    impl_tags={"family": "pytorch", "backend": "eager"},
    impl_func=torch_rotary,
)
```
