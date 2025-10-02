# Flash Attention Implementation

## GPU Info

```python id=nv
import subprocess

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
```

## Flash Attention Benchmark

```python id=benchmark outputs=attn.jsonl
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "torch",
#     "kernels-benchmark-tools",
# ]
#
# [tool.uv.sources]
# kernels-benchmark-tools = { git = "https://github.com/drbh/kernels-benchmark-tools.git", branch = "main" }
# ///
import torch
import sys
import os
import kernels_benchmark_tools as kbt


def torch_flash(q, k, v):
    qt, kt, vt = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        o = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt)
    return o.transpose(1, 2).contiguous()

kbt.add(
    "torch_flash_ma",
    torch_flash,
    tags={"family": "torch-sdpa", "backend": "FLASH", "compile": "max-autotune"},
)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "float32" if device == "cpu" else "bfloat16"

    # Flux-like workloads scaled down for CPU testing
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
        jsonl="attn.jsonl",
        reps=5,
        warmup=2,
        gen=kbt.attn.gen_qkv,
        ref=kbt.attn.ref_math,
        cmp=kbt.attn.cmp_allclose,
    )
    kbt.summarize(["attn.jsonl"])
```
