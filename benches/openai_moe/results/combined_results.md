---
title: "OpenAI-style MoE Benchmark - Combined Results"
author: "uvnote"
theme: "dark"
syntax_theme: "monokai"
show_line_numbers: true
collapse_code: false
---

# OpenAI-style MoE (Mixture of Experts) Benchmarks - Aggregated Results

This document combines benchmark results from multiple OpenAI-style MoE implementations.

## Combined Summary and Visualization

![artifact:latency.svg]

```python id=combine collapse-code=true needs=../impls/binned_torch.md:benchmark,../impls/gpt_oss_moe.md:benchmark outputs=latency.svg
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
    # "PyTorch OpenAI MoE": "UVNOTE_FILE_TORCH_OPENAI_MOE_BENCHMARK",
    "Binned PyTorch": "UVNOTE_FILE_BINNED_TORCH_BENCHMARK",
    "GptOssExperts": "UVNOTE_FILE_GPT_OSS_MOE_BENCHMARK",
}

# Generate combined results with visualization
generate_combined_results(
    cache_env_map=cache_env_map,
    output_filename="openai_moe.jsonl",
    svg_filename="latency.svg"
)
```
