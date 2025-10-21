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
#     "torch",
#     "kernels-benchmark-tools",
# ]
#
# [tool.uv.sources]
# kernels-benchmark-tools = { path = "/home/ubuntu/Projects/kernels-benchmarks-consolidated/tools", editable = true }
# ///
import torch
import sys
import kernels_benchmark_tools as kbt


def torch_swiglu(input_tensor):
    """PyTorch native SwiGLU implementation"""
    # Split input into two halves
    d = input_tensor.shape[-1] // 2
    x1 = input_tensor[..., :d]   # First half
    x2 = input_tensor[..., d:]   # Second half

    # SwiGLU: silu(x1) * x2
    return torch.nn.functional.silu(x1) * x2


# Register the implementation
kbt.add(
    "torch_swiglu",
    torch_swiglu,
    tags={"family": "torch", "backend": "native", "compile": "none"},
)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "float32" if device == "cpu" else "bfloat16"

    # Generate workloads - using a subset for faster testing
    if device == "cuda":
        wl = list(kbt.activation.llama_workloads(dtype=dtype))[:3]
    else:
        wl = list(kbt.activation.cpu_workloads(dtype=dtype))[:3]

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
