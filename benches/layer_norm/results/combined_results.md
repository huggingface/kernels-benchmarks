---
title: "LayerNorm Benchmark - Combined Results"
---

# LayerNorm Benchmarks - Aggregated Results

![artifact:latency.svg]

```python id=combine needs=../impls/torch_layer_norm.md:benchmark,../impls/hf_kernels_layer_norm.md:benchmark outputs=latency.svg
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "kernels-benchmark-tools", "matplotlib"]
# [tool.uv.sources]
# kernels-benchmark-tools = { path = "../../../tools", editable = true }
# ///
from kernels_benchmark_tools.core.visuals import generate_combined_results

cache_env_map = {
    "Torch LayerNorm": "UVNOTE_FILE_TORCH_LAYER_NORM_BENCHMARK",
    "HF Kernels LayerNorm": "UVNOTE_FILE_HF_KERNELS_LAYER_NORM_BENCHMARK",
}

generate_combined_results(
    cache_env_map=cache_env_map,
    output_filename="ln.jsonl",
    svg_filename="latency.svg",
    figure_id="layernorm"
)
```
