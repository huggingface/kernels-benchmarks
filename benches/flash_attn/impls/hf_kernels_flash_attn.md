---
on_github: huggingface/kernels-uvnotes
on_huggingface: kernels-community/flash-attn2
---

# HF Kernels - Flash Attention

## HuggingFace Kernels Flash Attention Benchmark

```python id=benchmark outputs=attn.jsonl
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
import os
import kernels_benchmark_tools as kbt
from kernels import get_kernel

hf_kernels_flash_attn = get_kernel("kernels-community/flash-attn")


def hf_flash_attention(query, key, value):
    """HuggingFace Kernels Flash Attention"""
    return hf_kernels_flash_attn.fwd(query, key, value, is_causal=False)[0]


kbt.add(
    "hf_kernels_flash_attn",
    hf_flash_attention,
    tags={"family": "hf-kernels", "backend": "flash-attn", "compile": "none"},
)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("HF Kernels Flash Attention requires CUDA - skipping benchmark")
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
        profile_trace=True
    )
    kbt.summarize(["attn.jsonl"])
```
