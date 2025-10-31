---
on_github: huggingface/kernels-benchmarks
---

# Binned PyTorch - OpenAI-style MoE

## GPU Info

```python id=nv
import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
```

## OpenAI-style MoE Benchmark (Binned PyTorch)

```python id=benchmark outputs=openai_moe.jsonl timeout=1200
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "torch==2.8.0",
#     "kernels-benchmark-tools",
# ]
#
# [tool.uv.sources]
# kernels-benchmark-tools = { path = "../../../../../tools", editable = true }
# ///
import torch
import sys
from kernels_benchmark_tools import KernelTypeEnum, run_benchmark


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
            flat_pos = indices[start + i]  # flattened (token, slot)
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
    E, K = routing_weights.shape[2], router_indices.shape[1]

    indices, _, bins, _ = sort_tokens_by_expert(router_indices, E)
    x = binned_gather(hidden_states.view(-1, H), indices, bins, expert_capacity, K)

    gate_up = torch.bmm(x, gate_up_proj) + gate_up_proj_bias[..., None, :]
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]

    # clamp to limit
    limit = 7.0
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)

    glu = gate * torch.sigmoid(gate * 1.702)
    x = (up + 1) * glu
    x = torch.bmm(x, down_proj) + down_proj_bias[..., None, :]

    # build routing weights aligned to (token, slot)
    flat_dense = routing_weights.view(-1, E)  # [B*S, E]
    flat_router = router_indices.view(-1, K)  # [B*S, K]
    selected = torch.gather(flat_dense, 1, flat_router).reshape(-1)  # [B*S*K]

    # scatter back
    y = binned_scatter(x, indices, selected, bins, expert_capacity, K)  # [B*S, H]

    return y.view(B, S, H)


def binned_torch_openai_moe(
    hidden_states,
    router_indices,
    routing_weights,
    gate_up_proj,
    gate_up_proj_bias,
    down_proj,
    down_proj_bias,
):
    """
    Binned PyTorch implementation of OpenAI-style MoE.
    Sorts tokens by expert assignment for more efficient batched processing.
    """
    B, S = hidden_states.shape[0], hidden_states.shape[1]
    K = router_indices.shape[1]

    # Set expert_capacity to a reasonable value (max tokens per expert)
    # Use 2x the average to handle imbalance
    expert_capacity = (B * S * K * 2) // routing_weights.shape[2]

    return binned_experts_ref(
        hidden_states,
        router_indices,
        routing_weights,
        gate_up_proj,
        gate_up_proj_bias,
        down_proj,
        down_proj_bias,
        expert_capacity,
    )


run_benchmark(
    kernel_type=KernelTypeEnum.OPENAI_MOE,
    impl_name="binned_torch",
    impl_tags={"family": "pytorch", "backend": "eager"},
    impl_func=binned_torch_openai_moe,
    dtype="float32",
)
```
