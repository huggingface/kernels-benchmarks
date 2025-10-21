---
on_github: huggingface/kernels-uvnotes
---


# Torch Compile Variants!

This file benchmarks Flash Attention with different torch.compile modes.

## Flash Attention with torch.compile(mode="default")

```python id=benchmark_default outputs=attn_default.jsonl
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
import os
import kernels_benchmark_tools as kbt


def torch_flash_base(q, k, v):
    qt, kt, vt = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        o = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt)
    return o.transpose(1, 2).contiguous()


# Compile with default mode
compiled_flash_default = torch.compile(torch_flash_base, mode="default", fullgraph=True, dynamic=False)

kbt.add(
    "torch_flash_compiled_default",
    compiled_flash_default,
    tags={"family": "torch-sdpa", "backend": "FLASH", "compile": "default"},
)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "float32" if device == "cpu" else "bfloat16"

    # Flux-like workloads
    base = 1024 if device == "cuda" else 512
    flux_sizes = (
        [128, 256, 320, 384, 448, 512] if device == "cuda" else [64, 128, 192, 256]
    )
    heads = 24 if device == "cuda" else 8
    head_dim = 128 if device == "cuda" else 64

    wl = []
    for L in flux_sizes:
        wl.append(
            {
                "name": f"flux_L{L}",
                "batch": 1,
                "seq_len": base + L,
                "heads": heads,
                "head_dim": head_dim,
                "dtype": dtype,
                "device": device,
                "seed": 0,
            }
        )

    kbt.run(
        wl,
        jsonl="attn_default.jsonl",
        reps=5,
        warmup=2,
        gen=kbt.attn.gen_qkv,
        ref=kbt.attn.ref_math,
        cmp=kbt.attn.cmp_allclose,
        profile_trace=True
    )
    kbt.summarize(["attn_default.jsonl"])
```

## Flash Attention with torch.compile(mode="max-autotune")

```python id=benchmark_max_autotune outputs=attn_max_autotune.jsonl
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
import os
import kernels_benchmark_tools as kbt


def torch_flash_base(q, k, v):
    qt, kt, vt = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        o = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt)
    return o.transpose(1, 2).contiguous()


# Compile with max-autotune mode
compiled_flash_max_autotune = torch.compile(torch_flash_base, mode="max-autotune", fullgraph=True, dynamic=False)

kbt.add(
    "torch_flash_compiled_max_autotune",
    compiled_flash_max_autotune,
    tags={"family": "torch-sdpa", "backend": "FLASH", "compile": "max-autotune"},
)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "float32" if device == "cpu" else "bfloat16"

    # Flux-like workloads
    base = 1024 if device == "cuda" else 512
    flux_sizes = (
        [128, 256, 320, 384, 448, 512] if device == "cuda" else [64, 128, 192, 256]
    )
    heads = 24 if device == "cuda" else 8
    head_dim = 128 if device == "cuda" else 64

    wl = []
    for L in flux_sizes:
        wl.append(
            {
                "name": f"flux_L{L}",
                "batch": 1,
                "seq_len": base + L,
                "heads": heads,
                "head_dim": head_dim,
                "dtype": dtype,
                "device": device,
                "seed": 0,
            }
        )

    kbt.run(
        wl,
        jsonl="attn_max_autotune.jsonl",
        reps=5,
        warmup=2,
        gen=kbt.attn.gen_qkv,
        ref=kbt.attn.ref_math,
        cmp=kbt.attn.cmp_allclose,
    )
    kbt.summarize(["attn_max_autotune.jsonl"])
```
