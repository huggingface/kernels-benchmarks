---
title: "LayerNorm Benchmark - Combined Results"
author: "uvnote"
theme: "dark"
syntax_theme: "monokai"
show_line_numbers: true
collapse_code: false
---

# LayerNorm Benchmarks - Aggregated Results

This document combines benchmark results from multiple LayerNorm implementations.

## Combined Summary and Visualization

![artifact:latency.svg]

```python id=combine collapse-code=true needs=../impls/torch_layer_norm.md:benchmark,../impls/hf_kernels_layer_norm.md:benchmark outputs=latency.svg
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
    "PyTorch LayerNorm": "UVNOTE_FILE_TORCH_LAYER_NORM_BENCHMARK",
    "HF Kernels LayerNorm": "UVNOTE_FILE_HF_KERNELS_LAYER_NORM_BENCHMARK",
}

# Generate combined results with visualization
generate_combined_results(
    cache_env_map=cache_env_map,
    output_filename="layer_norm.jsonl",
    svg_filename="latency.svg"
)
```
