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
import torch.nn.functional as F
import sys
import os
from typing import Tuple
import kernels_benchmark_tools as kbt


def sort_tokens_by_expert(
    router_indices: torch.Tensor, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort tokens by expert assignment for efficient batching."""
    flat_indices = router_indices.flatten()
    sorted_values, sorted_indices = torch.sort(flat_indices)
    tokens_per_expert = torch.bincount(sorted_values, minlength=num_experts)
    bins = torch.cumsum(tokens_per_expert, dim=0)
    return sorted_indices, sorted_values, bins, tokens_per_expert


def binned_gather(
    x: torch.Tensor,
    indices: torch.Tensor,
    bins: torch.Tensor,
    expert_capacity: int,
    top_k: int,
) -> torch.Tensor:
    """Gather tokens into expert bins."""
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


def binned_scatter(
    x: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    bins: torch.Tensor,
    expert_capacity: int,
    top_k: int,
) -> torch.Tensor:
    """Scatter expert outputs back to tokens."""
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


def advanced_binned_moe(
    hidden_states,
    router_indices,
    routing_weights,
    gate_up_proj,
    gate_up_proj_bias,
    down_proj,
    down_proj_bias,
):
    """Advanced binned MoE using gather-scatter operations for efficiency."""
    B, S, H = hidden_states.shape
    E, K = routing_weights.shape[2], router_indices.shape[1]
    expert_capacity = (B * S * K // E) + 16  # Add some buffer

    # Sort tokens by expert
    indices, _, bins, _ = sort_tokens_by_expert(router_indices, E)

    # Gather tokens into expert bins
    x = binned_gather(hidden_states.view(-1, H), indices, bins, expert_capacity, K)

    # Expert computation
    gate_up = torch.bmm(x, gate_up_proj)
    gate_up += gate_up_proj_bias[..., None, :]

    gate, up = gate_up[..., ::2], gate_up[..., 1::2]

    # Clamp to limit (common in MoE implementations)
    limit = 7.0
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)

    # GLU activation
    glu = gate * torch.sigmoid(gate * 1.702)
    x = (up + 1) * glu

    # Down projection
    x = torch.bmm(x, down_proj) + down_proj_bias[..., None, :]

    # Build routing weights aligned to (token, slot)
    flat_dense = routing_weights.view(-1, E)
    flat_router = router_indices.view(-1, K)
    selected = torch.gather(flat_dense, 1, flat_router).reshape(-1)

    # Scatter back to original token positions
    y = binned_scatter(x, indices, selected, bins, expert_capacity, K)

    return y.view(B, S, H)


def naive_moe(
    hidden_states,
    router_indices,
    routing_weights,
    gate_up_proj,
    gate_up_proj_bias,
    down_proj,
    down_proj_bias,
):
    """Naive MoE implementation - process each token individually."""
    B, S, H = hidden_states.shape
    E = gate_up_proj.shape[0]

    output = torch.zeros_like(hidden_states)

    for b in range(B):
        for s in range(S):
            token = hidden_states[b, s]  # [H]

            # Get top-k experts for this token
            top_experts = router_indices[b * S + s]  # [K]
            weights = routing_weights[b, s]  # [E]

            token_output = torch.zeros_like(token)

            for expert_idx in top_experts:
                # Expert computation
                gate_up = (
                    torch.matmul(token, gate_up_proj[expert_idx])
                    + gate_up_proj_bias[expert_idx]
                )
                gate, up = gate_up[::2], gate_up[1::2]

                # Apply activation and clamping
                gate = gate.clamp(max=7.0)
                up = up.clamp(-7.0, 7.0)
                glu = gate * torch.sigmoid(gate * 1.702)
                intermediate = (up + 1) * glu

                # Down projection
                expert_output = (
                    torch.matmul(intermediate, down_proj[expert_idx])
                    + down_proj_bias[expert_idx]
                )

                # Apply routing weight
                token_output += expert_output * weights[expert_idx]

            output[b, s] = token_output

    return output


def batched_moe(
    hidden_states,
    router_indices,
    routing_weights,
    gate_up_proj,
    gate_up_proj_bias,
    down_proj,
    down_proj_bias,
):
    """Batched MoE implementation - process all experts in parallel but only use top-k."""
    B, S, H = hidden_states.shape
    E, K = routing_weights.shape[2], router_indices.shape[1]

    hidden_flat = hidden_states.view(-1, H)
    output = torch.zeros_like(hidden_flat)

    # Process each token to respect top-k constraint
    routing_flat = routing_weights.view(-1, E)
    router_flat = router_indices.view(-1, K)

    for token_idx in range(hidden_flat.shape[0]):
        token = hidden_flat[token_idx]
        top_experts = router_flat[token_idx]
        weights = routing_flat[token_idx]

        token_output = torch.zeros_like(token)
        for expert_idx in top_experts:
            # Expert computation
            gate_up = (
                torch.matmul(token, gate_up_proj[expert_idx])
                + gate_up_proj_bias[expert_idx]
            )
            gate, up = gate_up[::2], gate_up[1::2]
            gate = gate.clamp(max=7.0)
            up = up.clamp(-7.0, 7.0)
            glu = gate * torch.sigmoid(gate * 1.702)
            intermediate = (up + 1) * glu
            expert_output = (
                torch.matmul(intermediate, down_proj[expert_idx])
                + down_proj_bias[expert_idx]
            )
            token_output += expert_output * weights[expert_idx]

        output[token_idx] = token_output

    return output.view(B, S, H)


def optimized_moe(
    hidden_states,
    router_indices,
    routing_weights,
    gate_up_proj,
    gate_up_proj_bias,
    down_proj,
    down_proj_bias,
):
    """Optimized MoE using expert-centric batching to reduce redundant computation."""
    B, S, H = hidden_states.shape
    E, K = routing_weights.shape[2], router_indices.shape[1]

    hidden_flat = hidden_states.view(-1, H)
    output = torch.zeros_like(hidden_flat)

    routing_flat = routing_weights.view(-1, E)
    router_flat = router_indices.view(-1, K)

    # Expert-centric approach: batch all tokens that use the same expert
    for expert_idx in range(E):
        # Find all tokens that use this expert
        expert_mask = (router_flat == expert_idx).any(dim=1)
        if not expert_mask.any():
            continue

        # Get tokens and their routing weights for this expert
        expert_tokens = hidden_flat[expert_mask]
        expert_weights = routing_flat[expert_mask, expert_idx]

        if expert_tokens.shape[0] == 0:
            continue

        # Batch process all tokens for this expert
        gate_up = (
            torch.matmul(expert_tokens, gate_up_proj[expert_idx])
            + gate_up_proj_bias[expert_idx]
        )
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(max=7.0)
        up = up.clamp(-7.0, 7.0)
        glu = gate * torch.sigmoid(gate * 1.702)
        intermediate = (up + 1) * glu
        expert_outputs = (
            torch.matmul(intermediate, down_proj[expert_idx])
            + down_proj_bias[expert_idx]
        )

        # Apply routing weights and accumulate
        weighted_outputs = expert_outputs * expert_weights.unsqueeze(-1)
        output[expert_mask] += weighted_outputs

    return output.view(B, S, H)


@torch.compile
def _compiled_moe_core(
    hidden_states,
    router_indices,
    routing_weights,
    gate_up_proj,
    gate_up_proj_bias,
    down_proj,
    down_proj_bias,
):
    """Core MoE computation optimized for torch.compile."""
    B, S, H = hidden_states.shape
    E, K = routing_weights.shape[2], router_indices.shape[1]

    hidden_flat = hidden_states.view(-1, H)
    output = torch.zeros_like(hidden_flat)

    routing_flat = routing_weights.view(-1, E)
    router_flat = router_indices.view(-1, K)

    # Use more compile-friendly operations
    for token_idx in range(hidden_flat.shape[0]):
        token = hidden_flat[token_idx]
        top_experts = router_flat[token_idx]
        weights = routing_flat[token_idx]

        token_output = torch.zeros_like(token)
        for expert_idx in top_experts:
            # Use @ operator for better compilation
            gate_up = token @ gate_up_proj[expert_idx] + gate_up_proj_bias[expert_idx]
            gate, up = gate_up[::2], gate_up[1::2]
            gate = torch.clamp(gate, max=7.0)
            up = torch.clamp(up, -7.0, 7.0)
            glu = gate * torch.sigmoid(gate * 1.702)
            intermediate = (up + 1) * glu
            expert_output = (
                intermediate @ down_proj[expert_idx] + down_proj_bias[expert_idx]
            )
            token_output += expert_output * weights[expert_idx]

        output[token_idx] = token_output

    return output.view(B, S, H)


def compiled_moe(
    hidden_states,
    router_indices,
    routing_weights,
    gate_up_proj,
    gate_up_proj_bias,
    down_proj,
    down_proj_bias,
):
    """MoE implementation using torch.compile for optimization."""
    return _compiled_moe_core(
        hidden_states,
        router_indices,
        routing_weights,
        gate_up_proj,
        gate_up_proj_bias,
        down_proj,
        down_proj_bias,
    )


def binned_moe(
    hidden_states,
    router_indices,
    routing_weights,
    gate_up_proj,
    gate_up_proj_bias,
    down_proj,
    down_proj_bias,
):
    """Advanced binned MoE using gather-scatter operations for efficiency."""
    return advanced_binned_moe(
        hidden_states,
        router_indices,
        routing_weights,
        gate_up_proj,
        gate_up_proj_bias,
        down_proj,
        down_proj_bias,
    )


# Register implementations
kbt.add(
    "naive_moe",
    naive_moe,
    tags={"family": "naive", "optimization": "none", "compile": "none"},
)
kbt.add(
    "batched_moe",
    batched_moe,
    tags={"family": "batched", "optimization": "sequential", "compile": "none"},
)
kbt.add(
    "optimized_moe",
    optimized_moe,
    tags={"family": "optimized", "optimization": "masking", "compile": "none"},
)
kbt.add(
    "compiled_moe",
    compiled_moe,
    tags={"family": "compiled", "optimization": "torch_compile", "compile": "default"},
)
kbt.add(
    "binned_moe",
    binned_moe,
    tags={
        "family": "advanced",
        "optimization": "binned_gather_scatter",
        "compile": "none",
    },
)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "float32" if device == "cpu" else "bfloat16"

    # Generate workloads based on device
    if device == "cuda":
        print(f"Running MoE benchmarks on {device} with {dtype}")
        # Use smaller Mixtral workloads for reasonable test time
        wl = [
            {
                "name": "mixtral_S512",
                "batch": 1,
                "seq_len": 512,
                "hidden_dim": 1024,  # Reduced from 4096
                "expert_dim": 2048,  # Reduced from 14336
                "num_experts": 8,
                "top_k": 2,
                "dtype": dtype,
                "device": device,
                "seed": 42,
            }
        ]
    else:
        print(f"Running MoE benchmarks on {device} with {dtype}")
        wl = list(kbt.moe.small_moe_workloads(dtype=dtype))
        # Take subset for faster testing
        wl = wl[:2]

    print(f"Testing {len(wl)} workloads")
    for w in wl:
        print(
            f"  - {w['name']}: {w['seq_len']}x{w['hidden_dim']}, {w['num_experts']} experts, top-{w['top_k']}"
        )

    kbt.run(
        wl,
        jsonl="moe.jsonl",
        reps=3,  # Fewer reps since MoE is more expensive
        warmup=1,
        gen=kbt.moe.gen_inputs,
        ref=kbt.moe.naive_moe_ref,
        cmp=kbt.moe.cmp_allclose,
    )

    # Generate summary and visualization
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    kbt.summarize(["moe.jsonl"])

    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATION")
    print("=" * 60)

    try:
        kbt.viz(["moe.jsonl"])
        print("Visualization saved as latency.png")
    except ImportError:
        print("Visualization requires matplotlib. Install with: uv add matplotlib")
    except Exception as e:
        print(f"Visualization failed: {e}")
