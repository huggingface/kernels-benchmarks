---
title: "Flash Attention Benchmark - Combined Results"
author: "uvnote"
theme: "dark"
syntax_theme: "monokai"
show_line_numbers: true
collapse_code: false
---

# Flash Attention Benchmarks - Aggregated Results

This document combines benchmark results from multiple Flash Attention implementations.

## Combined Summary and Visualization

![artifact:latency.svg]

```python id=combine collapse-code=true needs=../impls/flash_attention.md:benchmark,../impls/mem_efficient_attention.md:benchmark,../impls/xformers.md:benchmark,../impls/hf_kernels_flash_attn.md:benchmark,../impls/hf_kernels_flash_attn3.md:benchmark,../impls/sage_attention.md:benchmark,../impls/flash_attn_cute.md:benchmark outputs=latency.svg
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "torch==2.8.0",
#     "kernels-benchmark-tools",
#     "matplotlib",
# ]
#
# [tool.uv.sources]
# kernels-benchmark-tools = { path = "../../../../../tools", editable = true }
# ///
from kernels_benchmark_tools.core.visuals import generate_combined_results

# Map display names to uvnote environment variables
cache_env_map = {
    "Flash (PyTorch SDPA)": "UVNOTE_FILE_FLASH_ATTENTION_BENCHMARK",
    "MemEff (PyTorch SDPA)": "UVNOTE_FILE_MEM_EFFICIENT_ATTENTION_BENCHMARK",
    "xFormers": "UVNOTE_FILE_XFORMERS_BENCHMARK",
    "HF Kernels Flash Attn": "UVNOTE_FILE_HF_KERNELS_FLASH_ATTN_BENCHMARK",
    "HF Kernels Flash Attn3": "UVNOTE_FILE_HF_KERNELS_FLASH_ATTN3_BENCHMARK",
    "SageAttention": "UVNOTE_FILE_SAGE_ATTENTION_BENCHMARK",
    "Flash Attn CUTE": "UVNOTE_FILE_FLASH_ATTN_CUTE_BENCHMARK",
}

# Generate combined results with visualization
generate_combined_results(
    cache_env_map=cache_env_map,
    output_filename="attention.jsonl",
    svg_filename="latency.svg"
)
```
