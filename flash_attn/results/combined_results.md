---
title: "Flash Attention Benchmark - Combined Results"
author: "uvnote"
theme: "dark"
syntax_theme: "monokai"
show_line_numbers: true
collapse_code: false
custom_css: |
    #output-setup {
        overflow-x: auto;
    }
    .cell-output {
        overflow: scroll;
    }
    .cell-stdout {
        width: max-content;
        overflow: scroll;
    }
    .cell-stderr {
        width: max-content;
        overflow: scroll;
        max-height: 300px;
    }
---

# Flash Attention Benchmarks - Aggregated Results

This document combines benchmark results from multiple attention implementations
using cross-file dependencies.

## Combined Summary and Visualization

```python id=combine collapse-code=true needs=../impls/flash_attention.md:benchmark,../impls/math_attention.md:benchmark,../impls/mem_efficient_attention.md:benchmark,../impls/xformers.md:benchmark,../impls/compiled_variants.md:benchmark_default,../impls/compiled_variants.md:benchmark_max_autotune,../impls/hf_kernels_flash_attn.md:benchmark,../impls/hf_kernels_flash_attn3.md:benchmark outputs=latency.png
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "torch",
#     "kernels-benchmark-tools",
#     "matplotlib",
# ]
#
# [tool.uv.sources]
# kernels-benchmark-tools = { git = "https://github.com/drbh/kernels-benchmark-tools.git", branch = "main" }
# ///
import torch
import sys
import os
import kernels_benchmark_tools as kbt
from pathlib import Path

# Discover the upstream artifact directories from environment variables
cache_dirs = {
    "Flash (PyTorch SDPA)": os.environ.get('UVNOTE_FILE_FLASH_ATTENTION_BENCHMARK'),
    "MemEff (PyTorch SDPA)": os.environ.get('UVNOTE_FILE_MEM_EFFICIENT_ATTENTION_BENCHMARK'),
    "Flash Attn 2": os.environ.get('UVNOTE_FILE_FLASH_ATTN2_BENCHMARK'),
    "xFormers": os.environ.get('UVNOTE_FILE_XFORMERS_BENCHMARK'),
    "SageAttention": os.environ.get('UVNOTE_FILE_SAGE_ATTENTION_BENCHMARK'),
    "Compiled (default)": os.environ.get('UVNOTE_FILE_COMPILED_VARIANTS_BENCHMARK_DEFAULT'),
    "Compiled (max-autotune)": os.environ.get('UVNOTE_FILE_COMPILED_VARIANTS_BENCHMARK_MAX_AUTOTUNE'),
    "HF Kernels Flash Attn": os.environ.get('UVNOTE_FILE_HF_KERNELS_FLASH_ATTN_BENCHMARK'),
    "HF Kernels Flash Attn3": os.environ.get('UVNOTE_FILE_HF_KERNELS_FLASH_ATTN3_BENCHMARK'),
}

print("LOADING BENCHMARK DATA")
for name, cache_dir in cache_dirs.items():
    print(f"{name:30s}: {cache_dir}")
print()

# Collect all JSONL paths
all_paths = []
file_mapping = {
    "Flash (PyTorch SDPA)": "attn.jsonl",
    "MemEff (PyTorch SDPA)": "attn.jsonl",
    "Flash Attn 2": "attn.jsonl",
    "xFormers": "attn.jsonl",
    "SageAttention": "attn.jsonl",
    "Compiled (default)": "attn_default.jsonl",
    "Compiled (max-autotune)": "attn_max_autotune.jsonl",
    "HF Kernels Flash Attn": "attn.jsonl",
    "HF Kernels Flash Attn3": "attn.jsonl",
}

for name, cache_dir in cache_dirs.items():
    if cache_dir:
        jsonl_file = file_mapping[name]
        path = Path(cache_dir) / jsonl_file
        if path.exists() and path.stat().st_size > 0:
            all_paths.append(str(path))
            print(f"✓ Found {name}: {path}")
        else:
            print(f"⊘ Empty/Missing {name}: {path}")
    else:
        print(f"✗ No cache dir for {name}")

print()

if not all_paths:
    print("ERROR: No benchmark data files found!")
    sys.exit(1)

# Generate combined summary
print("COMBINED BENCHMARK SUMMARY")
print()

kbt.summarize(all_paths)

print()
print("GENERATING COMBINED VISUALIZATION")
print()

try:
    kbt.viz(all_paths)
    print("✓ Combined visualization saved as latency.png")
except ImportError as e:
    print(f"✗ Visualization requires matplotlib: {e}")
except Exception as e:
    print(f"✗ Visualization failed: {e}")

print()
print("ANALYSIS COMPLETE")
print(f"Total implementations analyzed: {len(all_paths)}")
print(f"\nImplementations included:")
for name, cache_dir in cache_dirs.items():
    if cache_dir:
        jsonl_file = file_mapping[name]
        path = Path(cache_dir) / jsonl_file
        if path.exists() and path.stat().st_size > 0:
            print(f"  ✓ {name}")
```
