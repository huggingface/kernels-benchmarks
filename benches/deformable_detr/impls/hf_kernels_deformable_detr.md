---
on_github: huggingface/kernels-uvnotes
---

# HF Kernels - Deformable DETR

## GPU Info

```python id=nv
import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
```

## Deformable DETR Multi-Scale Deformable Attention Benchmark

```python id=benchmark outputs=deformable_detr.jsonl
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

# Load the deformable DETR kernel
deformable_detr = get_kernel("kernels-community/deformable-detr")


def hf_kernels_deformable_detr(
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step=64
):
    """HuggingFace Kernels Deformable DETR Multi-Scale Deformable Attention"""
    return deformable_detr.ms_deform_attn_forward(
        value=value,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        sampling_loc=sampling_locations,
        attn_weight=attention_weights,
        im2col_step=im2col_step
    )


run_benchmark(
    kernel_type=KernelTypeEnum.DEFORMABLE_DETR,
    impl_name="hf_kernels_deformable_detr",
    impl_tags={"family": "hf-kernels", "backend": "cuda"},
    impl_func=hf_kernels_deformable_detr,
    dtype="float32",
)
```
