---
on_github: huggingface/kernels-uvnotes
---

# xFormers Memory Efficient Attention

## xFormers Benchmark

```python id=benchmark outputs=attn.jsonl
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "torch",
#     "kernels-benchmark-tools",
#     "xformers",
# ]
#
# [tool.uv.sources]
# kernels-benchmark-tools = { path = "/home/ubuntu/Projects/kernels-benchmarks-consolidated/tools", editable = true }
# ///
import torch
import sys
import os
import kernels_benchmark_tools as kbt
import xformers.ops as xops


def xformers_attention(q, k, v):
    """xFormers memory efficient attention"""
    # xFormers expects [batch, seq_len, heads, head_dim]
    return xops.memory_efficient_attention(q, k, v)


kbt.add(
    "xformers_meff",
    xformers_attention,
    tags={"family": "xformers", "backend": "memory_efficient", "compile": "none"},
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
