# How to Add a Benchmark

Add a new benchmark to the Kernels Benchmark Tools suite using **SwiGLU activation** as an example.

## Overview

Adding a benchmark involves three main steps:

1. **Add support to the tools library** - Create helper functions for your operation
2. **Create implementation benchmarks** - Write uvnote files that benchmark different implementations
3. **Run and aggregate results** - Execute benchmarks and combine results

> [!NOTE]
> **uvnote** files use `.md` extension but contain executable Python code with inline dependency declarations (PEP 723). They're run with `uvnote run <file>.md`.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Add Operation Support to Tools Library](#step-1-add-operation-support-to-tools-library)
3. [Step 2: Create Benchmark Directory Structure](#step-2-create-benchmark-directory-structure)
4. [Step 3: Write Implementation Benchmarks](#step-3-write-implementation-benchmarks)
5. [Step 4: Run Benchmarks](#step-4-run-benchmarks)
6. [Quick Start Checklist](#quick-start-checklist)

---

## Prerequisites

From project root (`kernels-benchmarks-consolidated/`):

- Install tools library: `cd tools && pip install -e . && cd ..`
- Have the kernel implementation you want to benchmark

---

## Step 1: Add Operation Support to Tools Library

Create a module in `tools/kernels_benchmark_tools/` with reusable functions for your operation.

### 1.1 Create the Operation Module

Create a new Python file in `tools/kernels_benchmark_tools/` named after the operation category.

**Example:** `tools/kernels_benchmark_tools/activation.py` for SwiGLU (an activation function)

### 1.2 Implement Required Functions

Each operation module needs four key components:

#### A. Input Generator (`gen_inputs`)

```python
from typing import Sequence
import torch

def gen_inputs(wl: dict) -> Sequence[torch.Tensor]:
    """Generate input tensors from workload spec."""
    torch.manual_seed(wl.get("seed", 0))

    # Convert dtype string to torch dtype
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(wl.get("dtype", "bfloat16"), torch.bfloat16)

    input_tensor = torch.randn(
        wl["num_tokens"],
        2 * wl["hidden_dim"],
        device=wl.get("device", "cuda"),
        dtype=dtype
    )

    return (input_tensor,)
```

#### B. Reference Implementation (`ref_<operation>`)

```python
def ref_swiglu(inputs: Sequence[torch.Tensor]) -> torch.Tensor:
    """Reference SwiGLU: silu(x[:d]) * x[d:]"""
    input_tensor, = inputs

    # Split input into two halves
    d = input_tensor.shape[-1] // 2
    x1 = input_tensor[..., :d]
    x2 = input_tensor[..., d:]

    # SwiGLU: silu(x1) * x2
    return torch.nn.functional.silu(x1) * x2
```

#### C. Comparison Function (`cmp_allclose`)

```python
def cmp_allclose(out: torch.Tensor, ref: torch.Tensor, rtol=None, atol=None) -> dict:
    """Compare output with reference (tolerances auto-adjust for dtype)."""
    from .core.tools import detailed_tensor_comparison

    result = detailed_tensor_comparison(
        out, ref, rtol=rtol, atol=atol, name="SwiGLU Activation"
    )
    result["ref"] = "swiglu_bfloat16"  # Label for this reference implementation
    return result
```

> [!NOTE]
> Passing `rtol=None, atol=None` automatically sets appropriate tolerances for bfloat16/float16/float32. Failures show detailed diagnostics.

#### D. Workload Generators

```python
from typing import Iterable

def llama_workloads(dtype="bfloat16") -> Iterable[dict]:
    """Generate LLaMA-style workloads for SwiGLU benchmarking."""
    for num_tokens in [512, 1024, 2048, 4096]:
        for hidden_dim in [4096, 8192, 11008]:
            yield {
                "name": f"llama_T{num_tokens}_D{hidden_dim}",
                "num_tokens": num_tokens,
                "hidden_dim": hidden_dim,
                "dtype": dtype,
                "device": "cuda",
                "seed": 0,
            }
```

### 1.3 Register the Module

Add your module to `tools/kernels_benchmark_tools/__init__.py`:

```python
from . import activation
```


---

## Step 2: Create Benchmark Directory Structure

```bash
mkdir -p benches/activation/impls
mkdir -p benches/activation/results
```

---

## Step 3: Write Implementation Benchmarks

Create `benches/activation/impls/hf_kernels_swiglu.md`:

````markdown
# Benchmark for HF Kernels SwiGLU Implementation

```python id=benchmark outputs=activation.jsonl
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "kernels-benchmark-tools", "kernels"]
# [tool.uv.sources]
# kernels-benchmark-tools = { path = "../../tools", editable = true }
# ///
import torch
import kernels_benchmark_tools as kbt
from kernels import get_kernel

activation = get_kernel("kernels-community/activation")

def hf_kernels_swiglu(input_tensor):
    """Wrapper for your kernel implementation"""
    hidden_dim = input_tensor.shape[-1] // 2
    out = torch.empty(input_tensor.shape[:-1] + (hidden_dim,),
                      dtype=input_tensor.dtype, device=input_tensor.device)
    return activation.silu_and_mul(out, input_tensor)

kbt.add("hf_kernels_swiglu", hf_kernels_swiglu,
        tags={"family": "hf-kernels", "backend": "triton"})

if __name__ == "__main__":
    wl = list(kbt.activation.llama_workloads(dtype="bfloat16"))
    kbt.run(wl, jsonl="activation.jsonl", reps=5, warmup=2,
            gen=kbt.activation.gen_inputs,
            ref=kbt.activation.ref_swiglu,
            cmp=kbt.activation.cmp_allclose)
    kbt.summarize(["activation.jsonl"])
```
````

**Pattern**: Write a wrapper function that takes inputs from `gen_inputs()`, calls your kernel, and returns output matching the reference. Register with `kbt.add()`, run with `kbt.run()`.

---

## Step 4: Run Benchmarks

### 4.1 Run Individual Benchmark

From project root:

```bash
uvnote run benches/activation/impls/hf_kernels_swiglu.md
```

### 4.2 Create Combined Results

Create `benches/activation/results/combined_results.md`:

````markdown
```python id=combine needs=../impls/hf_kernels_swiglu.md:benchmark,../impls/torch_swiglu.md:benchmark outputs=latency.svg
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "kernels-benchmark-tools", "matplotlib"]
# [tool.uv.sources]
# kernels-benchmark-tools = { path = "../../../tools", editable = true }
# ///
from kernels_benchmark_tools.core.visuals import generate_combined_results

# Map display names to uvnote environment variables
# uvnote sets these automatically based on filename and code block id
# Pattern: UVNOTE_FILE_<FILENAME>_<ID> (uppercase, hyphens to underscores)
# Example: hf_kernels_swiglu.md with id=benchmark → UVNOTE_FILE_HF_KERNELS_SWIGLU_BENCHMARK
cache_env_map = {
    "HF Kernels": "UVNOTE_FILE_HF_KERNELS_SWIGLU_BENCHMARK",
    "PyTorch": "UVNOTE_FILE_TORCH_SWIGLU_BENCHMARK",
}

generate_combined_results(cache_env_map, "activation.jsonl", "latency.svg")
```
````

Run from project root: `uvnote run benches/activation/results/combined_results.md`

---

## Quick Start Checklist

- [ ] Create `tools/kernels_benchmark_tools/<operation>.py` with: `gen_inputs`, `ref_<operation>`, `cmp_allclose`, workload generators
- [ ] Add `from . import <operation>` to `tools/kernels_benchmark_tools/__init__.py`
- [ ] Create directories: `benches/<operation>/impls/` and `benches/<operation>/results/`
- [ ] Write implementation benchmarks in `benches/<operation>/impls/*.md`
- [ ] Create `benches/<operation>/results/combined_results.md` using `generate_combined_results()`
- [ ] Run with `uvnote run`

---

## Tips and Best Practices

### Choosing Tolerances

Pass `rtol=None, atol=None` to use automatic tolerances adjusted for bfloat16/float16/float32 precision limits. Override only if needed.

### Workload Design

Use real-world model sizes (e.g., LLaMA dimensions) and test on relevant devices (CUDA/CPU).

### Debugging Failures

Comparison failures show detailed diagnostics: shape/dtype mismatches, NaN/Inf detection, worst mismatches, and suggestions.

### Performance

Use `warmup=2-5` and `reps=5-10` for CUDA kernels. Enable `profile_trace=True` for detailed analysis.
