---
on_github: huggingface/kernels-uvnotes
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
import kernels_benchmark_tools as kbt
from kernels import get_kernel

# Load the activation kernel
activation = get_kernel("kernels-community/activation")


def hf_kernels_swiglu(input_tensor):
    """HuggingFace Kernels SwiGLU implementation"""
    hidden_dim = input_tensor.shape[-1] // 2
    out_shape = input_tensor.shape[:-1] + (hidden_dim,)
    out = torch.empty(out_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    return activation.silu_and_mul(out, input_tensor)


# Register the implementation
kbt.add(
    "hf_kernels_swiglu",
    hf_kernels_swiglu,
    tags={"family": "hf-kernels", "backend": "triton", "compile": "none"},
)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("HF Kernels SwiGLU requires CUDA - skipping benchmark")
        sys.exit(0)

    dtype = "bfloat16"

    # Generate workloads - using a subset for faster testing
    wl = list(kbt.activation.llama_workloads(dtype=dtype))[:3]  # First 3 workloads

    print(f"Running SwiGLU benchmarks on {device} with {dtype}")
    print(f"Testing {len(wl)} workloads")

    # Run benchmark
    kbt.run(
        wl,
        jsonl="activation.jsonl",
        reps=5,
        warmup=2,
        gen=kbt.activation.gen_inputs,
        ref=kbt.activation.ref_swiglu,
        cmp=kbt.activation.cmp_allclose,
        profile_trace=True
    )

    kbt.summarize(["activation.jsonl"])
```
