---
title: "Causal Conv1D Benchmark - Combined Results"
author: "uvnote"
theme: "dark"
syntax_theme: "monokai"
show_line_numbers: true
collapse_code: false
platforms:
  - linux
---

# Causal Conv1D Benchmarks - Aggregated Results

This document combines benchmark results from multiple Causal Conv1D implementations.

## Combined Summary and Visualization

![artifact:latency.svg]

```python id=combine collapse-code=true needs=../impls/hf_kernels_causal_conv1d.md:benchmark,../impls/torch_causal_conv1d.md:benchmark outputs=latency.svg
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
    "HF Kernels Causal Conv1D": "UVNOTE_FILE_HF_KERNELS_CAUSAL_CONV1D_BENCHMARK",
    "PyTorch Causal Conv1D": "UVNOTE_FILE_TORCH_CAUSAL_CONV1D_BENCHMARK",
}

# Generate combined results with visualization
generate_combined_results(
    cache_env_map=cache_env_map,
    output_filename="causal_conv1d.jsonl",
    svg_filename="latency.svg"
)
```
