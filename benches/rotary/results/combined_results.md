---
title: "Rotary Position Embeddings Benchmark - Combined Results"
author: "uvnote"
theme: "dark"
syntax_theme: "monokai"
show_line_numbers: true
collapse_code: false
---

# Rotary Position Embeddings Benchmarks - Aggregated Results

This document combines benchmark results from multiple Rotary Position Embeddings implementations.

## Combined Summary and Visualization

![artifact:latency.svg]

```python id=combine collapse-code=true needs=../impls/hf_kernels_rotary.md:benchmark,../impls/torch_rotary.md:benchmark outputs=latency.svg
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
    "HF Kernels Rotary": "UVNOTE_FILE_HF_KERNELS_ROTARY_BENCHMARK",
    "PyTorch Rotary": "UVNOTE_FILE_TORCH_ROTARY_BENCHMARK",
}

# Generate combined results with visualization
generate_combined_results(
    cache_env_map=cache_env_map,
    output_filename="rotary.jsonl",
    svg_filename="latency.svg"
)
```
