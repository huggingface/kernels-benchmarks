---
title: "Flash Attention Benchmark - Combined Results"
---

# Flash Attention Benchmarks - Aggregated Results

![artifact:latency.svg]

```python id=combine needs=../impls/flash_attention.md:benchmark,../impls/mem_efficient_attention.md:benchmark,../impls/xformers.md:benchmark,../impls/compiled_variants.md:benchmark_default,../impls/compiled_variants.md:benchmark_max_autotune,../impls/hf_kernels_flash_attn.md:benchmark,../impls/hf_kernels_flash_attn3.md:benchmark outputs=latency.svg
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "kernels-benchmark-tools", "matplotlib"]
# [tool.uv.sources]
# kernels-benchmark-tools = { path = "../../../../../tools", editable = true }
# ///
from kernels_benchmark_tools.core.visuals import generate_combined_results

# Note: Flash attention has multiple implementations with different output files
# Some use attn.jsonl, compiled variants use attn_default.jsonl and attn_max_autotune.jsonl
cache_env_map = {
    "Flash (PyTorch SDPA)": "UVNOTE_FILE_FLASH_ATTENTION_BENCHMARK",
    "MemEff (PyTorch SDPA)": "UVNOTE_FILE_MEM_EFFICIENT_ATTENTION_BENCHMARK",
    "xFormers": "UVNOTE_FILE_XFORMERS_BENCHMARK",
    "Compiled (default)": "UVNOTE_FILE_COMPILED_VARIANTS_BENCHMARK_DEFAULT",
    "Compiled (max-autotune)": "UVNOTE_FILE_COMPILED_VARIANTS_BENCHMARK_MAX_AUTOTUNE",
    "HF Kernels Flash Attn": "UVNOTE_FILE_HF_KERNELS_FLASH_ATTN_BENCHMARK",
    "HF Kernels Flash Attn3": "UVNOTE_FILE_HF_KERNELS_FLASH_ATTN3_BENCHMARK",
}

# For flash attention, we need custom file mapping
import os
from pathlib import Path

file_mapping = {
    "Compiled (default)": "attn_default.jsonl",
    "Compiled (max-autotune)": "attn_max_autotune.jsonl",
}

# Collect paths with custom file names for compiled variants
all_paths = []
for name, env_var in cache_env_map.items():
    cache_dir = os.environ.get(env_var)
    if cache_dir:
        filename = file_mapping.get(name, "attn.jsonl")
        path = Path(cache_dir) / filename
        if path.exists() and path.stat().st_size > 0:
            all_paths.append(str(path))
            print(f"✓ Found {name}: {path}")
        else:
            print(f"⊘ Skipped {name}: {path}")
    else:
        print(f"✗ Missing {name}")

if not all_paths:
    print("ERROR: No benchmark data files found!")
    import sys
    sys.exit(1)

# Use the simplified visualization
from kernels_benchmark_tools.core import tools
from kernels_benchmark_tools.core.visuals import setup_svg_matplotlib, create_svg_with_tagging

setup_svg_matplotlib()
_orig_savefig, _orig_close = create_svg_with_tagging("latency.svg", "flash-attention")

try:
    print("\nCOMBINED BENCHMARK SUMMARY\n")
    tools.summarize(all_paths)

    print("\nGENERATING COMBINED VISUALIZATION\n")
    tools.viz(all_paths)

    import matplotlib.pyplot as plt
    plt.savefig("latency.svg")
    print("✓ SVG visualization ready!")
finally:
    plt.savefig = _orig_savefig
    plt.close = _orig_close
```
