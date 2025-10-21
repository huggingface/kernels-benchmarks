from __future__ import annotations
from typing import Sequence, Iterable
import torch
import torch.nn.functional as F


def _dtype(s: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[s]


def gen_inputs(wl: dict) -> Sequence[torch.Tensor]:
    """Generate MoE inputs: hidden_states, router_indices, routing_weights, and expert weights."""
    torch.manual_seed(int(wl.get("seed", 0)))

    B, S, H = wl["batch"], wl["seq_len"], wl["hidden_dim"]
    E, K = wl["num_experts"], wl["top_k"]
    dt = _dtype(wl.get("dtype", "bfloat16"))
    dev = wl.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Hidden states
    hidden_states = torch.randn(B, S, H, device=dev, dtype=dt) * 0.1

    # Router indices (top-k experts per token)
    router_indices = torch.randint(0, E, (B * S, K), device=dev)

    # Routing weights (softmax normalized)
    routing_weights = torch.rand(B, S, E, device=dev, dtype=dt)
    routing_weights = F.softmax(routing_weights, dim=-1)

    # Expert weights
    expert_dim = wl.get("expert_dim", H * 4)  # Typical MLP expansion

    # Gate-up projection: [E, H, 2*expert_dim]
    gate_up_proj = torch.randn(E, H, 2 * expert_dim, device=dev, dtype=dt) * 0.1
    gate_up_proj_bias = torch.randn(E, 2 * expert_dim, device=dev, dtype=dt) * 0.01

    # Down projection: [E, expert_dim, H]
    down_proj = torch.randn(E, expert_dim, H, device=dev, dtype=dt) * 0.1
    down_proj_bias = torch.randn(E, H, device=dev, dtype=dt) * 0.01

    return (
        hidden_states,
        router_indices,
        routing_weights,
        gate_up_proj,
        gate_up_proj_bias,
        down_proj,
        down_proj_bias,
    )


def naive_moe_ref(inputs: Sequence[torch.Tensor]) -> torch.Tensor:
    """Naive reference implementation for comparison - simple and correct."""
    (
        hidden_states,
        router_indices,
        routing_weights,
        gate_up_proj,
        gate_up_proj_bias,
        down_proj,
        down_proj_bias,
    ) = inputs

    # Convert to float32 for reference computation
    hidden_states = hidden_states.to(torch.float32)
    routing_weights = routing_weights.to(torch.float32)
    gate_up_proj = gate_up_proj.to(torch.float32)
    gate_up_proj_bias = gate_up_proj_bias.to(torch.float32)
    down_proj = down_proj.to(torch.float32)
    down_proj_bias = down_proj_bias.to(torch.float32)

    B, S, H = hidden_states.shape
    E, K = routing_weights.shape[2], router_indices.shape[1]

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

    return output.to(inputs[0].dtype)


def cmp_allclose(out: torch.Tensor, ref: torch.Tensor, rtol=1e-2, atol=1e-2) -> dict:
    """Compare MoE output with reference using allclose."""
    diff = (out - ref).abs()
    ok = torch.allclose(out, ref, rtol=rtol, atol=atol) and not (
        torch.isnan(out).any() or torch.isinf(out).any()
    )
    return {
        "ok": bool(ok),
        "rtol": rtol,
        "atol": atol,
        "absmax": float(diff.max().item()),
        "mae": float(diff.mean().item()),
        "mse": float(((out - ref) ** 2).mean().item()),
        "ref": "naive_moe_fp32",
    }


def mixtral_workloads(dtype="bfloat16") -> Iterable[dict]:
    """Generate Mixtral-style MoE workloads."""
    for seq_len in [512, 1024, 2048]:
        yield {
            "name": f"mixtral_S{seq_len}",
            "batch": 1,
            "seq_len": seq_len,
            "hidden_dim": 4096,
            "expert_dim": 14336,  # Mixtral expert size
            "num_experts": 8,
            "top_k": 2,
            "dtype": dtype,
            "device": "cuda",
            "seed": 42,
        }


def small_moe_workloads(dtype="float32") -> Iterable[dict]:
    """Generate smaller MoE workloads for testing/CPU."""
    for seq_len in [64, 128, 256]:
        for num_experts in [4, 8]:
            yield {
                "name": f"small_S{seq_len}_E{num_experts}",
                "batch": 1,
                "seq_len": seq_len,
                "hidden_dim": 512,
                "expert_dim": 1024,
                "num_experts": num_experts,
                "top_k": 2,
                "dtype": dtype,
                "device": "cpu",
                "seed": 42,
            }


def switch_transformer_workloads(dtype="bfloat16") -> Iterable[dict]:
    """Generate Switch Transformer style workloads (top-1)."""
    for seq_len in [512, 1024]:
        yield {
            "name": f"switch_S{seq_len}",
            "batch": 1,
            "seq_len": seq_len,
            "hidden_dim": 2048,
            "expert_dim": 5120,
            "num_experts": 16,
            "top_k": 1,  # Switch uses top-1
            "dtype": dtype,
            "device": "cuda",
            "seed": 42,
        }
