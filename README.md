# kernels benchmarks

> [!WARNING]
> This is an experimental repository and is subject to change. Benches may be unstables as a result of changes to the benchmarking tools. Pending v1.0.0 release.

This repo contains benchmarks for the kernels on the https://huggingface.co/kernels-community and https://github.com/huggingface/kernels-community. 


### Benchmark interface

Each bench has two parts:
1) the typed definition and reference in the `tools` folder, and 
2) the benchmark implementations in the `benches` folder.

A reference is defined by adding a new `KernelTypeEnum` value and corresponding file that includes the 3 standard functions: `gen_inputs`, `ref_impl`, and `cmp_allclose`. 

An implementation is defined by adding a new uvnote that implements a function that matches the benchmark interface. 

Then we can simply call `run_benchmark` with the kernel type to benchmark the function and create a output that we can use to generate reports.


**Example of benchmark implementation:**
`flash_attn/impls/hf_kernels_flash_attn`

```python
import sys

import torch
from kernels_benchmark_tools import KernelTypeEnum, run_benchmark
from kernels import get_kernel

# Load the flash attention kernel
hf_kernels_flash_attn = get_kernel("kernels-community/flash-attn2")

# Define the benchmark implementation function
def hf_flash_attention(query, key, value):
    return hf_kernels_flash_attn.fwd(query, key, value, is_causal=False)[0]

# Register and run the benchmark given the kernel type and input function
run_benchmark(
    kernel_type=KernelTypeEnum.ATTENTION,
    impl_name="hf_kernels_flash_attn",
    impl_tags={"family": "hf-kernels", "backend": "cuda"},
    impl_func=hf_flash_attention,
)
```

## Uvnote Reports

The final benchmark reports are generated using uvnote. This allows us to create rich markdown reports with embedded benchmark results; similar to Jupyter notebooks but with support for full customization of the output and more reproducibility (since we leverage python scripts which allow us to embed versioning info, etc).

Practically, in CI we run `uvnote build` on this repo to generate a aggregated report of all benchmarks. This reports has simple navigation to drill down into each implementation (that includes torch profiler traces) or at a kernel level to compare different implementations in a single view.

The final output of the aggregated report can be found on the Hugging Face Spaces at https://huggingface.co/spaces/kernels-community/kernels-benchmarks 

You can navigate to `<benchmark_name>/results/combined_results.html` to see an example of a aggregated benchmark report for a specific benchmark.

## Contributing

We more than welcome contributions to this repository! We always want more implementations of kernels to benchmark and compare, the more implementations the better developers can understand the tradeoffs of different kernels.