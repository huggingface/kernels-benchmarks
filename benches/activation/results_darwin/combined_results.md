---
title: "SwiGLU Activation Benchmark - Combined Results (macOS)"
author: "uvnote"
theme: "dark"
syntax_theme: "monokai"
show_line_numbers: true
collapse_code: false
platforms:
  - darwin
---

# SwiGLU Activation Benchmarks - Aggregated Results (macOS)

This document combines benchmark results from SwiGLU activation implementations on macOS.

## Combined Summary and Visualization

![artifact:latency.svg]

```python id=combine collapse-code=true needs=../impls/torch_swiglu_darwin.md:benchmark outputs=latency.svg
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "torch==2.8.0",
#     "kernels-benchmark-tools",
#     "matplotlib"
# ]
#
# [tool.uv.sources]
# kernels-benchmark-tools = { path = "../../../../../tools", editable = true }
# ///
from kernels_benchmark_tools.core.visuals import generate_combined_results

# Map display names to uvnote environment variables
cache_env_map = {
    "PyTorch SwiGLU (macOS)": "UVNOTE_FILE_TORCH_SWIGLU_DARWIN_BENCHMARK",
}

# Generate combined results with visualization
generate_combined_results(
    cache_env_map=cache_env_map,
    output_filename="activation.jsonl",
    svg_filename="latency.svg"
)
```
