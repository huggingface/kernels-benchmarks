---
on_github: huggingface/kernels-uvnotes
---

# SageAttention Implementation

## SageAttention Benchmark (INT8 Quantized)

```python id=benchmark outputs=attn.jsonl
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "torch==2.8.0",
#     "kernels",
#     "kernels-benchmark-tools",
#     "sageattention",
# ]
#
# [tool.uv.sources]
# kernels-benchmark-tools = { path = "../../../../../tools", editable = true }
# ///
import torch
import sys
import os
import kernels_benchmark_tools as kbt
# from sageattention import sageattn_qk_int8_pv_fp16_cuda


# def sage_attention(q, k, v):
#     """SageAttention with INT8 Q/K quantization and FP16 P/V"""
#     return sageattn_qk_int8_pv_fp16_cuda(q, k, v, tensor_layout="NHD")

from kernels import get_kernel

hf_kernels_sage_attn = get_kernel("kernels-community/sage_attention")


def sage_attention(query, key, value):
    """HuggingFace Kernels Flash Attention"""
    return hf_kernels_sage_attn.fwd(query, key, value, is_causal=False)[0]

kbt.add(
    "sage_int8_fp16",
    sage_attention,
    tags={"family": "sageattention", "backend": "int8_fp16_cuda", "compile": "none"},
)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("SageAttention requires CUDA - skipping benchmark")
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
