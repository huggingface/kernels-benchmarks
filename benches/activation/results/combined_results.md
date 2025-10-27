---
title: "SwiGLU Activation Benchmark - Combined Results"
author: "uvnote"
theme: "dark"
syntax_theme: "monokai"
show_line_numbers: true
collapse_code: false
---

# SwiGLU Activation Benchmarks - Aggregated Results

This document combines benchmark results from multiple SwiGLU activation implementations.

## Combined Summary and Visualization

![artifact:latency.svg]

```python id=combine collapse-code=true needs=../impls/hf_kernels_swiglu.md:benchmark,../impls/torch_swiglu.md:benchmark outputs=latency.svg
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
    "HF Kernels SwiGLU": "UVNOTE_FILE_HF_KERNELS_SWIGLU_BENCHMARK",
    "PyTorch SwiGLU": "UVNOTE_FILE_TORCH_SWIGLU_BENCHMARK",
    # "Compiled SwiGLU": "UVNOTE_FILE_COMPILED_SWIGLU_BENCHMARK",
}

# Generate combined results with visualization
generate_combined_results(
    cache_env_map=cache_env_map,
    output_filename="activation.jsonl",
    svg_filename="latency.svg"
)
```
