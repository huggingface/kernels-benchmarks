# HF Kernels - Flash Attention 3

## HuggingFace Kernels Flash Attention 3 Benchmark

```python id=benchmark outputs=attn.jsonl
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "torch",
#     "kernels-benchmark-tools",
#     "kernels",
# ]
#
# [tool.uv.sources]
# kernels-benchmark-tools = { git = "https://github.com/drbh/kernels-benchmark-tools.git", branch = "main" }
# ///
import torch
import sys
import os
import kernels_benchmark_tools as kbt
from kernels import get_kernel

hf_kernels_flash_attn3 = get_kernel("kernels-community/flash-attn3")


def hf_flash_attention3(query, key, value):
    return hf_kernels_flash_attn3.flash_attn_func(query, key, value, causal=False)[0]


kbt.add(
    "hf_kernels_flash_attn3",
    hf_flash_attention3,
    tags={"family": "hf-kernels", "backend": "flash-attn3", "compile": "none"},
)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("HF Kernels Flash Attention 3 requires CUDA - skipping benchmark")
        sys.exit(0)

    dtype = "bfloat16"

    # Flux-like workloads
    base = 1024
    flux_sizes = [128, 256, 320, 384, 448, 512]
    heads = 24
    head_dim = 128

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
