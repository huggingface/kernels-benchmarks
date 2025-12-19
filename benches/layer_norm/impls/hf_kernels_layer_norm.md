---
on_github: huggingface/kernels-benchmarks
on_huggingface: kernels-community/layer-norm
platforms:
  - linux
---

# HF Kernels LayerNorm Implementation

Based on kernels-community `layer-norm` kernel.

## LayerNorm Benchmark (HF Kernels)

```python id=benchmark outputs=layer_norm.jsonl
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "torch==2.8.0",
#     "kernels",
#     "kernels-benchmark-tools",
# ]
#
# [tool.uv.sources]
# kernels-benchmark-tools = { path = "../../../../../tools", editable = true }
# ///
import torch
import sys
from kernels_benchmark_tools import KernelTypeEnum, run_benchmark
from kernels import get_kernel

# Load the layer norm kernel
layer_norm_kernel = get_kernel("kernels-community/layer-norm")


def hf_kernels_layer_norm(x, weight, bias, eps: float = 1e-5):
    B, S, D = x.shape
    # The kernel expects [N, D] input; support beta (bias) if provided.
    out = layer_norm_kernel.dropout_add_ln_fwd(
        input=x.view(-1, D),
        gamma=weight,
        beta=bias,
        rowscale=None,
        colscale=None,
        x0_subset=None,
        z_subset=None,
        dropout_p=0.0,
        epsilon=eps,
        rowscale_const=1.0,
        z_numrows=S,
        gen=None,
        residual_in_fp32=False,
        is_rms_norm=False,
    )[0].view(B, S, D)
    return out


run_benchmark(
    kernel_type=KernelTypeEnum.LAYER_NORM,
    impl_name="hf_kernels_layer_norm",
    impl_tags={"family": "hf-kernels", "repo": "kernels-community/layer-norm", "op": "layer_norm"},
    impl_func=hf_kernels_layer_norm,
)
```
