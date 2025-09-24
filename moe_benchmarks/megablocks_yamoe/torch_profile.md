---
title: "Compare Yamoe and Binned MoE Implementations"
author: "David Holtz"
theme: "dark"
syntax_theme: "monokai"
show_line_numbers: true
collapse_code: false
custom_css: |
    .cell-stderr { 
        display: block; 
    }
    .minimap { 
        display: none !important; 
    }
    .file-explorer { 
        display: none !important; 
    }
    .cell-code { 
        max-height: 400px; 
        overflow: auto; 
    }
---

```python id=utils deps=torch,numpy
"""Simple utilities for running the models."""
import torch

def to_dtype(dtype_str: str):
    """Convert string to torch dtype."""
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    return torch.float32

def tensor_stats(t: torch.Tensor) -> str:
    """Generate stats string for a tensor."""
    return (f"shape={tuple(t.shape)}, "
            f"dtype={t.dtype}, "
            f"device={t.device}, "
            f"mean={t.mean().item():.6f}, "
            f"std={t.std().item():.6f}")

def set_seed(seed: int):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

```python id=bench_utils deps=torch,numpy
"""Reusable benchmarking utilities for performance testing."""
import time
import numpy as np
from contextlib import contextmanager
from typing import Callable, Dict, Tuple, Any, Optional
import torch

def to_dtype(dtype_str: str):
    """Convert string to torch dtype."""
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    return torch.float32

def _sync(device: str):
    """Synchronize device if CUDA."""
    if device == "cuda":
        torch.cuda.synchronize()

def _compute_stats(times_s, tokens: Optional[int] = None) -> Dict[str, float]:
    """Compute comprehensive latency and throughput statistics."""
    lat_ms = np.array([t * 1000.0 for t in times_s])
    lat_ms_sorted = np.sort(lat_ms)
    n = len(lat_ms)
    
    stats = {
        "avg_ms": np.mean(lat_ms),
        "min_ms": np.min(lat_ms),
        "max_ms": np.max(lat_ms),
        "std_ms": np.std(lat_ms),
        "p50_ms": np.percentile(lat_ms, 50),
        "p95_ms": np.percentile(lat_ms, 95),
        "p99_ms": np.percentile(lat_ms, 99),
        "num_iters": n
    }
    
    if tokens is not None and n > 0:
        avg_s = np.mean(times_s)
        stats["tokens_per_s"] = tokens / avg_s if avg_s > 0 else float("inf")
        stats["throughput_variance"] = np.std([tokens / t for t in times_s if t > 0])
    
    return stats

def _format_timing_stats(stats: Dict[str, float], tokens: Optional[int] = None) -> str:
    """Format timing statistics for display."""
    lines = [
        "\n━━━━━━━━━━━━━━━━━━━━ Benchmark Results ━━━━━━━━━━━━━━━━━━━━",
        f"Iterations: {stats.get('num_iters', 0)}",
        "\nLatency Statistics:",
        f"  Average: {stats['avg_ms']:.3f} ms",
        f"  Min:     {stats['min_ms']:.3f} ms",
        f"  Max:     {stats['max_ms']:.3f} ms", 
        f"  Std Dev: {stats['std_ms']:.3f} ms",
        "\nPercentiles:",
        f"  P50 (median): {stats['p50_ms']:.3f} ms",
        f"  P95:          {stats['p95_ms']:.3f} ms",
        f"  P99:          {stats['p99_ms']:.3f} ms",
    ]
    
    if tokens is not None and 'tokens_per_s' in stats:
        lines.extend([
            "\nThroughput:",
            f"  Tokens/sec: {stats['tokens_per_s']:.1f}",
            f"  Std Dev:    {stats.get('throughput_variance', 0):.1f}",
        ])
    
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)

def _bench_engine(
    call: Callable[[], Any], *, warmup: int, iters: int, device: str, dtype, input_gen: Callable[[], Any] = None
) -> Tuple[Any, list]:
    """Core benchmarking engine with warmup and timing."""
    use_autocast = device == "cuda" and dtype in (torch.float16, torch.bfloat16)

    # Warmup phase
    print(f"\nWarming up ({warmup} iterations)...")
    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    if input_gen is not None:
                        _ = call(input_gen())
                    else:
                        _ = call()
            else:
                if input_gen is not None:
                    _ = call(input_gen())
                else:
                    _ = call()
        _sync(device)

    # Measurement phase
    print(f"Benchmarking ({iters} iterations)...")
    times_s = []
    last = None
    with torch.inference_mode():
        for i in range(max(1, iters)):
            start = time.perf_counter()
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    if input_gen is not None:
                        last = call(input_gen())
                    else:
                        last = call()
            else:
                if input_gen is not None:
                    last = call(input_gen())
                else:
                    last = call()
            _sync(device)
            end = time.perf_counter()
            times_s.append(end - start)

            # Progress indicator every 20% of iterations
            if i > 0 and i % max(1, iters // 5) == 0:
                pct = (i / iters) * 100
                avg_so_far = np.mean(times_s[:i]) * 1000
                print(f"  Progress: {pct:.0f}% complete (avg: {avg_so_far:.3f} ms)")

    return last, times_s

def tensor_stats(t: torch.Tensor) -> str:
    """Generate comprehensive stats string for a tensor."""
    return (f"shape={tuple(t.shape)}, "
            f"dtype={t.dtype}, "
            f"device={t.device}, "
            f"range=[{t.min().item():.6f}, {t.max().item():.6f}], "
            f"mean={t.mean().item():.6f}, "
            f"std={t.std().item():.6f}, "
            f"norm={t.norm().item():.6f}")

@contextmanager
def bench_context(
    *, warmup: int = 25, iters: int = 100, device: str = "cuda", dtype=torch.float32, tokens: Optional[int] = None, verbose: bool = True, save_json: Optional[str] = None, vary_inputs: bool = True
):
    """Context that yields a runner: runner(fn, *args, **kwargs) -> (result, stats).

    If vary_inputs=True, the first argument should be a base tensor that will be varied each iteration
    by adding a small deterministic increment to prevent caching artifacts.
    """

    def runner(fn: Callable[..., Any], *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
        # Log configuration
        if verbose:
            print(f"\n┌─ Benchmark Configuration ─────────────────────────────┐")
            # print(f"│ Device: {device:<15} Dtype: {dtype}              │")
            print(f"│ Warmup: {warmup:<15} Iters: {iters}              │")
            if tokens:
                print(f"│ Tokens: {tokens}                                        │")
            if vary_inputs:
                print(f"│ Input Variation: Enabled (prevents caching artifacts)  │")
            print(f"└────────────────────────────────────────────────────────┘")

        # Set up input generation
        input_gen = None
        if vary_inputs and args and isinstance(args[0], torch.Tensor):
            base_input = args[0].clone()
            iteration_counter = [0]  # Use list for mutable closure

            def generate_varied_input():
                """Generate input tensor varied by iteration to prevent caching."""
                # Add small deterministic increment: 0.001 * iteration_number
                varied_input = base_input + (iteration_counter[0] * 0.001)
                iteration_counter[0] += 1
                return varied_input

            input_gen = generate_varied_input
            call = lambda x: fn(x, *args[1:], **kwargs)

            # Log base input stats
            if verbose:
                print(f"\nBase Input: {tensor_stats(base_input)}")
                print(f"Input Variation: +{0.001:.3f} * iteration (deterministic)")
        else:
            # Legacy mode - static inputs
            call = lambda: fn(*args, **kwargs)
            if verbose and args and isinstance(args[0], torch.Tensor):
                print(f"\nInput: {tensor_stats(args[0])}")

        result, times_s = _bench_engine(call, warmup=warmup, iters=iters, device=device, dtype=dtype, input_gen=input_gen)

        # Log output if it's a tensor or tuple with tensors
        if verbose:
            print("\nOutput tensors:")
            if isinstance(result, torch.Tensor):
                print(f"  Primary: {tensor_stats(result)}")
            elif isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], torch.Tensor):
                print(f"  Primary: {tensor_stats(result[0])}")
                if len(result) > 1:
                    if isinstance(result[1], torch.Tensor):
                        print(f"  Auxiliary: {tensor_stats(result[1])}")
                    else:
                        print(f"  Auxiliary: {type(result[1]).__name__}")

        # Compute and display statistics
        stats = _compute_stats(times_s, tokens=tokens)
        if verbose:
            print(_format_timing_stats(stats, tokens))

        # Save to JSON if requested
        if save_json:
            import json
            json_data = {
                "implementation": save_json.replace(".json", ""),
                "config": {
                    "warmup": warmup,
                    "iters": iters,
                    "device": str(device),  # Convert device to string
                    "dtype": str(dtype),
                    "tokens": tokens,
                    "vary_inputs": vary_inputs
                },
                "stats": stats,
                "output_sum": float(result[0].sum().item()) if isinstance(result, tuple) and len(result) > 0 else float(result.sum().item()) if isinstance(result, torch.Tensor) else None
            }
            with open(save_json, 'w') as f:
                json.dump(json_data, f, indent=2)
            if verbose:
                print(f"\nSaved benchmark results to {save_json}")

        return result, stats

    yield runner

def set_seed(seed: int):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```


This notebook benchmarks multiple MoE implementations with varied inputs across iterations to prevent unrealistic caching artifacts and measure true performance characteristics.

```python id=config deps=torch,numpy
"""Shared configuration for both implementations."""
import torch

# Model configuration
NUM_EXPERTS = 128
HIDDEN_SIZE = 1152
INTERMEDIATE_SIZE = 3072
TOP_K = 4

# Input configuration
BATCH_SIZE = 1
SEQ_LEN = 100
DTYPE = "float32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Seeds for reproducibility
WEIGHT_SEED = 999
EXPERT_SEED = 777
INPUT_SEED = 123
GENERAL_SEED = 42
```


```python id=save_data depends=config deps=torch,numpy outputs=router_weight.pt,router_bias.pt,gate_up_proj.pt,gate_up_proj_bias.pt,down_proj.pt,down_proj_bias.pt
"""
Generate deterministic shared weights once and save as artifacts so
both implementations load identical parameters.
"""
import torch
from config import NUM_EXPERTS, HIDDEN_SIZE, WEIGHT_SEED, EXPERT_SEED

def save_shared_weights():
    # Router: Kaiming uniform as used by both, bias zeros
    torch.manual_seed(WEIGHT_SEED)
    router_weight = torch.empty(NUM_EXPERTS, HIDDEN_SIZE)
    torch.nn.init.kaiming_uniform_(router_weight)
    router_bias = torch.zeros(NUM_EXPERTS)

    # Experts: normal(0, 0.02), biases zeros
    torch.manual_seed(EXPERT_SEED)
    gate_up_proj = torch.empty(NUM_EXPERTS, HIDDEN_SIZE, 2 * HIDDEN_SIZE).normal_(mean=0.0, std=0.02)
    gate_up_proj_bias = torch.zeros(NUM_EXPERTS, 2 * HIDDEN_SIZE)
    down_proj = torch.empty(NUM_EXPERTS, HIDDEN_SIZE, HIDDEN_SIZE).normal_(mean=0.0, std=0.02)
    down_proj_bias = torch.zeros(NUM_EXPERTS, HIDDEN_SIZE)

    # Save artifacts
    torch.save(router_weight, 'router_weight.pt')
    torch.save(router_bias, 'router_bias.pt')
    torch.save(gate_up_proj, 'gate_up_proj.pt')
    torch.save(gate_up_proj_bias, 'gate_up_proj_bias.pt')
    torch.save(down_proj, 'down_proj.pt')
    torch.save(down_proj_bias, 'down_proj_bias.pt')

    print("Saved shared weights to artifacts")
    print(f"Router weight sum: {router_weight.sum().item():.6f}")
    print(f"Gate/up sum: {gate_up_proj.sum().item():.6f}")
    print(f"Down sum: {down_proj.sum().item():.6f}")

save_shared_weights()
```


## Yamoe Implementation

This section runs the Yamoe MoE implementation with optimized Triton kernels.

```python id=yamoe_run depends=config,save_data,bench_utils deps=torch,kernels,numpy
import torch
from torch import nn
from torch.nn import functional as F
from kernels import get_kernel, get_local_kernel
from bench_utils import to_dtype, tensor_stats, set_seed, bench_context
from config import (
    NUM_EXPERTS, HIDDEN_SIZE, TOP_K,
    BATCH_SIZE, SEQ_LEN, DTYPE, DEVICE,
    WEIGHT_SEED, EXPERT_SEED, INPUT_SEED, GENERAL_SEED
)
from pathlib import Path
import os

# Discover the upstream artifact directory from env
data_dir = os.environ.get('UVNOTE_INPUT_SAVE_DATA', '.')
print(f"Loading weights from: {data_dir}")

router_weight = torch.load(Path(data_dir) / 'router_weight.pt')
router_bias = torch.load(Path(data_dir) / 'router_bias.pt')
gate_up_proj = torch.load(Path(data_dir) / 'gate_up_proj.pt')
gate_up_proj_bias = torch.load(Path(data_dir) / 'gate_up_proj_bias.pt')
down_proj = torch.load(Path(data_dir) / 'down_proj.pt')
down_proj_bias = torch.load(Path(data_dir) / 'down_proj_bias.pt')

print("Loaded shared weights from artifacts")
print(f"Router weight sum: {router_weight.sum().item():.6f}")
print(f"Gate/up sum: {gate_up_proj.sum().item():.6f}")
print(f"Down sum: {down_proj.sum().item():.6f}")

class YamoeRouter(nn.Module):
    def __init__(self, router_weight, router_bias):
        super().__init__()
        self.top_k = TOP_K
        self.num_experts = NUM_EXPERTS
        self.hidden_dim = HIDDEN_SIZE
        self.weight = nn.Parameter(router_weight.clone())
        self.bias = nn.Parameter(router_bias.clone())

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight, self.bias)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        return router_scores, router_indices

def ceil_div(a, b):
    return (a + b - 1) // b

class YamoeMoEMLP(nn.Module):
    def __init__(self, router_weight, router_bias, gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias):
        super().__init__()
        self.router = YamoeRouter(router_weight, router_bias)
        self.num_experts = NUM_EXPERTS
        self.hidden_size = HIDDEN_SIZE
        self.top_k = TOP_K
        
        # Load Yamoe kernel
        # self.yamoe = get_local_kernel(Path("/home/ubuntu/Projects/yamoe/result"), "yamoe")
        self.yamoe = get_kernel("drbh/yamoe", revision="v0.2.0")

        # Expert weights - use the loaded weights
        self.gate_up_proj = nn.Parameter(gate_up_proj.clone())
        self.gate_up_proj_bias = nn.Parameter(gate_up_proj_bias.clone())
        self.down_proj = nn.Parameter(down_proj.clone())
        self.down_proj_bias = nn.Parameter(down_proj_bias.clone())

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Get routing decisions
        routing_weights, router_indices = self.router(hidden_states)
        
        # Reshape for Yamoe kernel
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        routing_weights_flat = routing_weights.view(-1, self.num_experts)
        expert_capacity = ceil_div(batch_size * self.top_k, self.num_experts)

        # Call Yamoe optimized kernel
        output = self.yamoe.experts(
            hidden_states_flat,
            router_indices,
            routing_weights_flat,
            self.gate_up_proj,
            self.gate_up_proj_bias,
            self.down_proj,
            self.down_proj_bias,
            expert_capacity,
            self.num_experts,
            self.top_k,
        )
        
        # Reshape output back
        output = output.view(batch_size, seq_len, hidden_dim)
        
        return output, routing_weights

# Run the model
set_seed(GENERAL_SEED)

device = torch.device(DEVICE if DEVICE == "cuda" else "cuda")
dtype = to_dtype(DTYPE)

print("\n=== Yamoe Implementation ===")
# Initialize model with loaded weights
model = YamoeMoEMLP(
    router_weight.to(device),
    router_bias.to(device),
    gate_up_proj.to(device),
    gate_up_proj_bias.to(device),
    down_proj.to(device),
    down_proj_bias.to(device)
).to(device=device)

print(f"Router weight sum: {model.router.weight.sum().item():.6f}")
print(f"Gate/up proj sum: {model.gate_up_proj.sum().item():.6f}")
print(f"Down proj sum: {model.down_proj.sum().item():.6f}")

# Generate input
set_seed(INPUT_SEED)
x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=device, dtype=dtype) * 0.1

# Benchmark the model with varied inputs to prevent caching artifacts
tokens = BATCH_SIZE * SEQ_LEN
with bench_context(warmup=10, iters=50, device=device, dtype=dtype, tokens=tokens, save_json="yamoe_results.json", vary_inputs=True) as bench:
    output, stats = bench(model, x)
    print(f"\nOutput sum: {output[0].sum().item():.6f}")
```



## Binned Implementation

This section runs the binned implementation that manually handles token gathering/scattering.

```python id=binned_run depends=config,bench_utils,save_data deps=torch,numpy
import torch
from torch import nn
from torch.nn import functional as F
from bench_utils import to_dtype, tensor_stats, set_seed, bench_context
from config import (
    NUM_EXPERTS, HIDDEN_SIZE, TOP_K,
    BATCH_SIZE, SEQ_LEN, DTYPE, DEVICE,
    WEIGHT_SEED, EXPERT_SEED, INPUT_SEED, GENERAL_SEED
)
from pathlib import Path
import os

# Discover the upstream artifact directory from env
data_dir = os.environ.get('UVNOTE_INPUT_SAVE_DATA', '.')

router_weight = torch.load(Path(data_dir) / 'router_weight.pt')
router_bias = torch.load(Path(data_dir) / 'router_bias.pt')
gate_up_proj = torch.load(Path(data_dir) / 'gate_up_proj.pt')
gate_up_proj_bias = torch.load(Path(data_dir) / 'gate_up_proj_bias.pt')
down_proj = torch.load(Path(data_dir) / 'down_proj.pt')
down_proj_bias = torch.load(Path(data_dir) / 'down_proj_bias.pt')

print("Loaded shared weights from artifacts")
print(f"Router weight sum: {router_weight.sum().item():.6f}")
print(f"Gate/up sum: {gate_up_proj.sum().item():.6f}")
print(f"Down sum: {down_proj.sum().item():.6f}")

def binned_gather(x, indices, bins, expert_capacity, top_k):
    E, H = bins.shape[0], x.shape[1]
    out = torch.zeros((E, expert_capacity, H), device=x.device, dtype=x.dtype)
    for e in range(E):
        start = 0 if e == 0 else bins[e - 1]
        end = bins[e]
        n = min(end - start, expert_capacity)
        for i in range(n):
            flat_pos = indices[start + i]
            tok = flat_pos // top_k
            out[e, i] = x[tok]
    return out

def binned_scatter(x, indices, weights, bins, expert_capacity, top_k):
    E, C, H = x.shape
    N = indices.shape[0] // top_k
    out = torch.zeros((N, top_k, H), dtype=x.dtype, device=x.device)
    for e in range(E):
        start = 0 if e == 0 else bins[e - 1]
        end = bins[e]
        n = end - start
        if n == 0:
            continue
        take = min(n, expert_capacity)
        for i in range(take):
            flat_pos = indices[start + i]
            tok = flat_pos // top_k
            slot = flat_pos % top_k
            scale = weights[flat_pos] if weights is not None else 1.0
            out[tok, slot] = x[e, i] * scale
    return out.sum(dim=1)

def sort_tokens_by_expert(router_indices, num_experts):
    flat_indices = router_indices.flatten()
    sorted_values, sorted_indices = torch.sort(flat_indices)
    tokens_per_expert = torch.bincount(sorted_values, minlength=num_experts)
    bins = torch.cumsum(tokens_per_expert, dim=0)
    return sorted_indices, sorted_values, bins, tokens_per_expert

def binned_experts_ref(
    hidden_states,
    router_indices,
    routing_weights,
    gate_up_proj,
    gate_up_proj_bias,
    down_proj,
    down_proj_bias,
    expert_capacity,
):
    B, S, H = hidden_states.shape
    E, K = routing_weights.shape[1], router_indices.shape[1]

    indices, _, bins, _ = sort_tokens_by_expert(router_indices, E)
    x = binned_gather(hidden_states.view(-1, H), indices, bins, expert_capacity, K)

    gate_up = torch.bmm(x, gate_up_proj) 
    gate_up += gate_up_proj_bias[..., None, :]

    gate, up = gate_up[..., ::2], gate_up[..., 1::2]

    # clamp to limit
    limit = 7.0
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)

    glu = gate * torch.sigmoid(gate * 1.702)
    x = (up + 1) * glu
    x = torch.bmm(x, down_proj) + down_proj_bias[..., None, :]

    # build routing weights aligned to (token, slot)
    flat_dense = routing_weights.view(-1, E)
    flat_router = router_indices.view(-1, K)
    selected = torch.gather(flat_dense, 1, flat_router).reshape(-1)

    # scatter back
    y = binned_scatter(x, indices, selected, bins, expert_capacity, K)

    return y.view(B, S, H)

class BinnedRouter(nn.Module):
    def __init__(self, router_weight, router_bias):
        super().__init__()
        self.top_k = TOP_K
        self.num_experts = NUM_EXPERTS
        self.hidden_dim = HIDDEN_SIZE
        self.weight = nn.Parameter(router_weight.clone())
        self.bias = nn.Parameter(router_bias.clone())

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight, self.bias)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        return router_scores, router_indices

def ceil_div(a, b):
    return (a + b - 1) // b

class BinnedMoEMLP(nn.Module):
    def __init__(self, router_weight, router_bias, gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias):
        super().__init__()
        self.router = BinnedRouter(router_weight, router_bias)
        self.num_experts = NUM_EXPERTS
        self.hidden_size = HIDDEN_SIZE
        self.top_k = TOP_K
        
        # Expert weights - use the loaded weights
        self.gate_up_proj = nn.Parameter(gate_up_proj.clone())
        self.gate_up_proj_bias = nn.Parameter(gate_up_proj_bias.clone())
        self.down_proj = nn.Parameter(down_proj.clone())
        self.down_proj_bias = nn.Parameter(down_proj_bias.clone())

    def forward(self, hidden_states):
        router_scores, router_indices = self.router(hidden_states)
        batch_size = hidden_states.shape[0]
        expert_capacity = ceil_div(batch_size * self.top_k, self.num_experts)

        output = binned_experts_ref(
            hidden_states,
            router_indices,
            router_scores,
            self.gate_up_proj,
            self.gate_up_proj_bias,
            self.down_proj,
            self.down_proj_bias,
            expert_capacity,
        )
        
        return output, router_scores

# Run the model
set_seed(GENERAL_SEED)

device = torch.device(DEVICE)
dtype = to_dtype(DTYPE)

print("\n=== Binned Implementation ===")
# Initialize model with loaded weights
model = BinnedMoEMLP(
    router_weight.to(device),
    router_bias.to(device),
    gate_up_proj.to(device),
    gate_up_proj_bias.to(device),
    down_proj.to(device),
    down_proj_bias.to(device)
).to(device=device)

print(f"Router weight sum: {model.router.weight.sum().item():.6f}")
print(f"Gate/up proj sum: {model.gate_up_proj.sum().item():.6f}")
print(f"Down proj sum: {model.down_proj.sum().item():.6f}")

# Generate the same input as Yamoe
set_seed(INPUT_SEED)
x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=device, dtype=dtype) * 0.1

# Benchmark the model with varied inputs to prevent caching artifacts
tokens = BATCH_SIZE * SEQ_LEN
with bench_context(warmup=10, iters=50, device=device, dtype=dtype, tokens=tokens, save_json="binned_results.json", vary_inputs=True) as bench:
    output, stats = bench(model, x)
    print(f"\nOutput sum: {output[0].sum().item():.6f}")
```

## GPT-OSS Implementation

This section runs the GPT-OSS MoE implementation with manual expert loop handling.

```python id=gptoss_run depends=config,bench_utils,save_data deps=torch,numpy
import torch
from torch import nn
from torch.nn import functional as F
from bench_utils import to_dtype, tensor_stats, set_seed, bench_context
from config import (
    NUM_EXPERTS, HIDDEN_SIZE, TOP_K,
    BATCH_SIZE, SEQ_LEN, DTYPE, DEVICE,
    WEIGHT_SEED, EXPERT_SEED, INPUT_SEED, GENERAL_SEED
)
from pathlib import Path
import os

# Discover the upstream artifact directory from env
data_dir = os.environ.get('UVNOTE_INPUT_SAVE_DATA', '.')

router_weight = torch.load(Path(data_dir) / 'router_weight.pt')
router_bias = torch.load(Path(data_dir) / 'router_bias.pt')
gate_up_proj = torch.load(Path(data_dir) / 'gate_up_proj.pt')
gate_up_proj_bias = torch.load(Path(data_dir) / 'gate_up_proj_bias.pt')
down_proj = torch.load(Path(data_dir) / 'down_proj.pt')
down_proj_bias = torch.load(Path(data_dir) / 'down_proj_bias.pt')

print("Loaded shared weights from artifacts")
print(f"Router weight sum: {router_weight.sum().item():.6f}")
print(f"Gate/up sum: {gate_up_proj.sum().item():.6f}")
print(f"Down sum: {down_proj.sum().item():.6f}")

class GptOssRouter(nn.Module):
    def __init__(self, router_weight, router_bias):
        super().__init__()
        self.top_k = TOP_K
        self.num_experts = NUM_EXPERTS
        self.hidden_dim = HIDDEN_SIZE
        self.weight = nn.Parameter(router_weight.clone())
        self.bias = nn.Parameter(router_bias.clone())

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight, self.bias)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        return router_scores, router_indices

class GptOssExperts(nn.Module):
    def __init__(self, gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias):
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.hidden_size = HIDDEN_SIZE
        self.expert_dim = self.hidden_size
        self.gate_up_proj = nn.Parameter(gate_up_proj.clone())
        self.gate_up_proj_bias = nn.Parameter(gate_up_proj_bias.clone())
        self.down_proj = nn.Parameter(down_proj.clone())
        self.down_proj_bias = nn.Parameter(down_proj_bias.clone())
        self.alpha = 1.702
        self.limit = 7.0

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        num_experts = routing_weights.shape[1]
        
        if hidden_states.device.type == "cpu" or self.training:
            next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            
            for expert_idx in expert_hit[:]:
                expert_idx = expert_idx[0]
                with torch.no_grad():
                    _, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                gate_up = current_state @ self.gate_up_proj[expert_idx] + self.gate_up_proj_bias[expert_idx]
                gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                gate = gate.clamp(min=None, max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
                glu = gate * torch.sigmoid(gate * self.alpha)
                gated_output = (up + 1) * glu
                out = gated_output @ self.down_proj[expert_idx] + self.down_proj_bias[expert_idx]
                weighted_output = out * routing_weights[token_idx, expert_idx, None]
                next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
            next_states = next_states.view(batch_size, -1, self.hidden_size)
        else:
            hidden_states = hidden_states.repeat(num_experts, 1)
            hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)
            gate_up = torch.bmm(hidden_states, self.gate_up_proj) + self.gate_up_proj_bias[..., None, :]
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)
            next_states = torch.bmm(((up + 1) * glu), self.down_proj)
            next_states = next_states + self.down_proj_bias[..., None, :]
            next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
            next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
            next_states = next_states.sum(dim=0)
        return next_states

class GptOssMoEMLP(nn.Module):
    def __init__(self, router_weight, router_bias, gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias):
        super().__init__()
        self.router = GptOssRouter(router_weight, router_bias)
        self.experts = GptOssExperts(gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias)

    def forward(self, hidden_states):
        router_scores, router_indices = self.router(hidden_states)
        routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
        return routed_out, router_scores

# Run the model
set_seed(GENERAL_SEED)

device = torch.device(DEVICE)
dtype = to_dtype(DTYPE)

print("\n=== GPT-OSS Implementation ===")
# Initialize model with loaded weights
model = GptOssMoEMLP(
    router_weight.to(device),
    router_bias.to(device),
    gate_up_proj.to(device),
    gate_up_proj_bias.to(device),
    down_proj.to(device),
    down_proj_bias.to(device)
).to(device=device)

print(f"Router weight sum: {model.router.weight.sum().item():.6f}")
print(f"Gate/up proj sum: {model.experts.gate_up_proj.sum().item():.6f}")
print(f"Down proj sum: {model.experts.down_proj.sum().item():.6f}")

# Generate the same input as other implementations
set_seed(INPUT_SEED)
x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=device, dtype=dtype) * 0.1

# Benchmark the model with varied inputs to prevent caching artifacts
tokens = BATCH_SIZE * SEQ_LEN
with bench_context(warmup=10, iters=50, device=device, dtype=dtype, tokens=tokens, save_json="gptoss_results.json", vary_inputs=True) as bench:
    output, stats = bench(model, x)
    print(f"\nOutput sum: {output[0].sum().item():.6f}")
```

## GPT-OSS Implementation (Training Mode)

This section runs the GPT-OSS MoE implementation with training mode enabled to force the expert loop path.

```python id=gptoss_training_run depends=config,bench_utils,save_data deps=torch,numpy
import torch
from torch import nn
from torch.nn import functional as F
from bench_utils import to_dtype, tensor_stats, set_seed, bench_context
from config import (
    NUM_EXPERTS, HIDDEN_SIZE, TOP_K,
    BATCH_SIZE, SEQ_LEN, DTYPE, DEVICE,
    WEIGHT_SEED, EXPERT_SEED, INPUT_SEED, GENERAL_SEED
)
from pathlib import Path
import os

# Discover the upstream artifact directory from env
data_dir = os.environ.get('UVNOTE_INPUT_SAVE_DATA', '.')

router_weight = torch.load(Path(data_dir) / 'router_weight.pt')
router_bias = torch.load(Path(data_dir) / 'router_bias.pt')
gate_up_proj = torch.load(Path(data_dir) / 'gate_up_proj.pt')
gate_up_proj_bias = torch.load(Path(data_dir) / 'gate_up_proj_bias.pt')
down_proj = torch.load(Path(data_dir) / 'down_proj.pt')
down_proj_bias = torch.load(Path(data_dir) / 'down_proj_bias.pt')

print("Loaded shared weights from artifacts")
print(f"Router weight sum: {router_weight.sum().item():.6f}")
print(f"Gate/up sum: {gate_up_proj.sum().item():.6f}")
print(f"Down sum: {down_proj.sum().item():.6f}")

class GptOssTrainingRouter(nn.Module):
    def __init__(self, router_weight, router_bias):
        super().__init__()
        self.top_k = TOP_K
        self.num_experts = NUM_EXPERTS
        self.hidden_dim = HIDDEN_SIZE
        self.weight = nn.Parameter(router_weight.clone())
        self.bias = nn.Parameter(router_bias.clone())

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight, self.bias)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        return router_scores, router_indices

class GptOssTrainingExperts(nn.Module):
    def __init__(self, gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias):
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.hidden_size = HIDDEN_SIZE
        self.expert_dim = self.hidden_size
        self.gate_up_proj = nn.Parameter(gate_up_proj.clone())
        self.gate_up_proj_bias = nn.Parameter(gate_up_proj_bias.clone())
        self.down_proj = nn.Parameter(down_proj.clone())
        self.down_proj_bias = nn.Parameter(down_proj_bias.clone())
        self.alpha = 1.702
        self.limit = 7.0

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        num_experts = routing_weights.shape[1]
        
        # Force training mode path (expert loop instead of batched)
        next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        
        for expert_idx in expert_hit[:]:
            expert_idx = expert_idx[0]
            with torch.no_grad():
                _, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up = current_state @ self.gate_up_proj[expert_idx] + self.gate_up_proj_bias[expert_idx]
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)
            gated_output = (up + 1) * glu
            out = gated_output @ self.down_proj[expert_idx] + self.down_proj_bias[expert_idx]
            weighted_output = out * routing_weights[token_idx, expert_idx, None]
            next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
        next_states = next_states.view(batch_size, -1, self.hidden_size)
        return next_states

class GptOssTrainingMoEMLP(nn.Module):
    def __init__(self, router_weight, router_bias, gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias):
        super().__init__()
        self.router = GptOssTrainingRouter(router_weight, router_bias)
        self.experts = GptOssTrainingExperts(gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias)

    def forward(self, hidden_states):
        router_scores, router_indices = self.router(hidden_states)
        routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
        return routed_out, router_scores

# Run the model
set_seed(GENERAL_SEED)

device = torch.device(DEVICE)
dtype = to_dtype(DTYPE)

print("\n=== GPT-OSS Implementation (Training Mode - Expert Loop) ===")
# Initialize model with loaded weights and force training mode
model = GptOssTrainingMoEMLP(
    router_weight.to(device),
    router_bias.to(device),
    gate_up_proj.to(device),
    gate_up_proj_bias.to(device),
    down_proj.to(device),
    down_proj_bias.to(device)
).to(device=device)

# Set to training mode to force expert loop path
model.train()

print(f"Router weight sum: {model.router.weight.sum().item():.6f}")
print(f"Gate/up proj sum: {model.experts.gate_up_proj.sum().item():.6f}")
print(f"Down proj sum: {model.experts.down_proj.sum().item():.6f}")
print(f"Model training mode: {model.training}")

# Generate the same input as other implementations
set_seed(INPUT_SEED)
x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=device, dtype=dtype) * 0.1

# Benchmark the model with varied inputs to prevent caching artifacts
tokens = BATCH_SIZE * SEQ_LEN
with bench_context(warmup=10, iters=50, device=device, dtype=dtype, tokens=tokens, save_json="gptoss_training_results.json", vary_inputs=True) as bench:
    output, stats = bench(model, x)
    print(f"\nOutput sum: {output[0].sum().item():.6f}")
```


## MegaBlocks Implementation

This section runs the MegaBlocks MoE implementation with optimized kernels from the Hugging Face hub.

```python id=megablocks_run depends=config,bench_utils,save_data deps=torch,numpy,kernels
import torch
from torch import nn
from torch.nn import functional as F
from kernels import get_kernel, get_local_kernel
from bench_utils import to_dtype, tensor_stats, set_seed, bench_context
from config import (
    NUM_EXPERTS, HIDDEN_SIZE, TOP_K,
    BATCH_SIZE, SEQ_LEN, DTYPE, DEVICE,
    WEIGHT_SEED, EXPERT_SEED, INPUT_SEED, GENERAL_SEED
)
from pathlib import Path
from collections import namedtuple
import os

# Discover the upstream artifact directory from env
data_dir = os.environ.get('UVNOTE_INPUT_SAVE_DATA', '.')

print(f"Loading weights from: {data_dir}")

router_weight = torch.load(Path(data_dir) / 'router_weight.pt')
router_bias = torch.load(Path(data_dir) / 'router_bias.pt')
gate_up_proj = torch.load(Path(data_dir) / 'gate_up_proj.pt')
gate_up_proj_bias = torch.load(Path(data_dir) / 'gate_up_proj_bias.pt')
down_proj = torch.load(Path(data_dir) / 'down_proj.pt')
down_proj_bias = torch.load(Path(data_dir) / 'down_proj_bias.pt')

print("Loaded shared weights from artifacts")
print(f"Router weight sum: {router_weight.sum().item():.6f}")
print(f"Gate/up sum: {gate_up_proj.sum().item():.6f}")
print(f"Down sum: {down_proj.sum().item():.6f}")

def build_megablocks_model(device: torch.device):
    # Download optimized kernels from the Hugging Face hub
    megablocks = get_kernel("kernels-community/megablocks", revision="v0.0.2")
    model = megablocks.layers.MegaBlocksMoeMLP()

    # Create attribute container for expert weights
    model.experts = namedtuple(
        "Experts", ["gate_up_proj", "gate_up_proj_bias", "down_proj", "down_proj_bias", "hidden_size"]
    )

    # Use loaded router weights for consistency
    model.router = torch.nn.Linear(HIDDEN_SIZE, NUM_EXPERTS, device=device)
    with torch.no_grad():
        model.router.weight.copy_(router_weight)
        model.router.bias.copy_(router_bias)

    # Attach loaded expert weights to the experts container
    e = model.experts
    e.alpha = 1.702
    e.capacity_factor = 32
    e.gate_up_proj = torch.nn.Parameter(gate_up_proj.clone().to(device))
    e.gate_up_proj_bias = torch.nn.Parameter(gate_up_proj_bias.clone().to(device))
    e.down_proj = torch.nn.Parameter(down_proj.clone().to(device))
    e.down_proj_bias = torch.nn.Parameter(down_proj_bias.clone().to(device))
    e.hidden_size = HIDDEN_SIZE
    
    # Log weight statistics for comparison
    print(f"[MegaBlocks] Router weight sum: {model.router.weight.sum().item():.6f}")
    print(f"[MegaBlocks] Gate/up projection shape: {tuple(e.gate_up_proj.shape)}, sum: {e.gate_up_proj.sum().item():.6f}")
    print(f"[MegaBlocks] Down projection shape: {tuple(e.down_proj.shape)}, sum: {e.down_proj.sum().item():.6f}")

    return model

# Create a wrapper to match the interface of other implementations
class MegaBlocksMoEWrapper(nn.Module):
    def __init__(self, megablocks_model):
        super().__init__()
        self.model = megablocks_model
        
    def forward(self, hidden_states):
        # MegaBlocks expects input in the format (batch, seq_len, hidden_dim)
        output, dummy_routing_weights = self.model(hidden_states)
        return output, dummy_routing_weights

# Run the model
set_seed(GENERAL_SEED)

device = torch.device(DEVICE)
dtype = to_dtype(DTYPE)

print("\n=== MegaBlocks Implementation ===")
# Build MegaBlocks model with loaded weights
megablocks_model = build_megablocks_model(device)
model = MegaBlocksMoEWrapper(megablocks_model).to(device=device)

# Generate the same input as other implementations
set_seed(INPUT_SEED)
x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=device, dtype=dtype) * 0.1

# Benchmark the model with varied inputs to prevent caching artifacts
tokens = BATCH_SIZE * SEQ_LEN
with bench_context(warmup=10, iters=50, device=device, dtype=dtype, tokens=tokens, save_json="megablocks_results.json", vary_inputs=True) as bench:
    output, stats = bench(model, x)
    print(f"\nOutput sum: {output[0].sum().item():.6f}")
```

## Performance Visualization

This section reads all benchmark results and creates a comprehensive performance comparison chart.

```python id=visualization depends=yamoe_run,binned_run,gptoss_run,gptoss_training_run,megablocks_run deps=matplotlib
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

# List of expected result files
yamoe_dir = os.environ.get('UVNOTE_INPUT_YAMOE_RUN', '.')
binned_dir = os.environ.get('UVNOTE_INPUT_BINNED_RUN', '.')
gptoss_dir = os.environ.get('UVNOTE_INPUT_GPTOSS_RUN', '.')
gptoss_training_dir = os.environ.get('UVNOTE_INPUT_GPTOSS_TRAINING_RUN', '.')
megablocks_dir = os.environ.get('UVNOTE_INPUT_MEGABLOCKS_RUN', '.')

result_files = [
    Path(yamoe_dir) / "yamoe_results.json",
    Path(binned_dir) / "binned_results.json", 
    Path(gptoss_dir) / "gptoss_results.json",
    Path(gptoss_training_dir) / "gptoss_training_results.json",
    Path(megablocks_dir) / "megablocks_results.json"
]

# Load all benchmark results
results = {}
for file in result_files:
    if Path(file).exists():
        with open(file, 'r') as f:
            data = json.load(f)
            results[data['implementation']] = data
        print(f"Loaded {file}")
    else:
        print(f"Missing {file}")

if not results:
    print("No benchmark results found. Run the benchmark cells first.")
else:
    # Extract data for plotting
    implementations = list(results.keys())
    avg_latencies = [results[impl]['stats']['avg_ms'] for impl in implementations]
    p95_latencies = [results[impl]['stats']['p95_ms'] for impl in implementations]
    throughputs = [results[impl]['stats'].get('tokens_per_s', 0) for impl in implementations]
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('MoE Implementation Performance Comparison', fontsize=16, fontweight='bold')
    
    # Colors for each implementation
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'][:len(implementations)]
    
    # 1. Average Latency Chart
    bars1 = ax1.bar(implementations, avg_latencies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Average Latency', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Latency (ms)', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, avg_latencies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_latencies)*0.01,
                f'{val:.2f}ms', ha='center', va='bottom', fontweight='bold')
    
    # 2. P95 Latency Chart
    bars2 = ax2.bar(implementations, p95_latencies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_title('95th Percentile Latency', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Latency (ms)', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars2, p95_latencies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(p95_latencies)*0.01,
                f'{val:.2f}ms', ha='center', va='bottom', fontweight='bold')
    
    # 3. Throughput Chart
    bars3 = ax3.bar(implementations, throughputs, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax3.set_title('Throughput', fontweight='bold', fontsize=14)
    ax3.set_ylabel('Tokens/sec', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars3, throughputs):
        if val > 0:  # Only show label if throughput was calculated
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(throughputs)*0.01,
                    f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("moe_performance_comparison.png", dpi=300)
    
    # Print summary table
    print("\nPerformance Summary:")
    print(f"{'Implementation':<30} {'Avg (ms)':<12} {'P95 (ms)':<12} {'Tokens/sec':<12} {'Relative Speed':<15}")
    print("-"*80)
    
    # Sort by average latency for relative speed calculation
    sorted_results = sorted(results.items(), key=lambda x: x[1]['stats']['avg_ms'])
    fastest_latency = sorted_results[0][1]['stats']['avg_ms']
    
    for impl, data in sorted_results:
        avg_ms = data['stats']['avg_ms']
        p95_ms = data['stats']['p95_ms']
        tokens_s = data['stats'].get('tokens_per_s', 0)
        relative_speed = fastest_latency / avg_ms
        
        print(f"{impl:<30} {avg_ms:>8.2f}    {p95_ms:>8.2f}    {tokens_s:>8.0f}      {relative_speed:>6.2f}x")
    
    print(f"\nFastest: {sorted_results[0][0]} ({sorted_results[0][1]['stats']['avg_ms']:.2f}ms avg)")
    if len(sorted_results) > 1:
        print(f"Slowest: {sorted_results[-1][0]} ({sorted_results[-1][1]['stats']['avg_ms']:.2f}ms avg)")
        speedup = sorted_results[-1][1]['stats']['avg_ms'] / sorted_results[0][1]['stats']['avg_ms']
        print(f"Max Speedup: {speedup:.1f}x")
```
