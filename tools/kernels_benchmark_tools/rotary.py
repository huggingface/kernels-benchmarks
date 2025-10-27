from __future__ import annotations
from typing import Sequence, Iterable
import torch


def gen_inputs(wl: dict) -> Sequence[torch.Tensor]:
    torch.manual_seed(int(wl.get("seed", 0)))
    batch = wl["batch"]
    seqlen = wl["seqlen"]
    num_heads = wl["num_heads"]
    head_dim = wl["head_dim"]
    rotary_dim = wl.get("rotary_dim", head_dim // 2)

    dt = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(wl.get("dtype", "bfloat16"))
    dev = wl.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Generate query and key tensors
    query = torch.randn(batch, seqlen, num_heads, head_dim, device=dev, dtype=dt)
    key = torch.randn(batch, seqlen, num_heads, head_dim, device=dev, dtype=dt)

    # Generate cos and sin for rotary position embeddings
    cos = torch.randn(seqlen, 1, rotary_dim, device=dev, dtype=dt)
    sin = torch.randn(seqlen, 1, rotary_dim, device=dev, dtype=dt)

    return (query, key, cos, sin)


def apply_rotary_torch(
    x1: torch.Tensor,
    x2: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    conj: bool = False,
):
    """Reference rotary implementation."""
    if not conj:
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
    else:
        out1 = x1 * cos + x2 * sin
        out2 = -x1 * sin + x2 * cos
    return out1, out2


def ref_impl(
    inputs: Sequence[torch.Tensor], conj: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation of rotary position embeddings."""
    query, key, cos, sin = inputs
    rotary_dim = cos.shape[-1]

    # Clone inputs to avoid modifying them
    q_out = query.clone()
    k_out = key.clone()

    # Apply rotation to query
    q1 = q_out[..., :rotary_dim]
    q2 = q_out[..., rotary_dim : 2 * rotary_dim]
    q_out_1, q_out_2 = apply_rotary_torch(q1, q2, cos, sin, conj)
    q_out[..., :rotary_dim] = q_out_1
    q_out[..., rotary_dim : 2 * rotary_dim] = q_out_2

    # Apply rotation to key
    k1 = k_out[..., :rotary_dim]
    k2 = k_out[..., rotary_dim : 2 * rotary_dim]
    k_out_1, k_out_2 = apply_rotary_torch(k1, k2, cos, sin, conj)
    k_out[..., :rotary_dim] = k_out_1
    k_out[..., rotary_dim : 2 * rotary_dim] = k_out_2

    return q_out, k_out


def cmp_allclose(
    out: tuple[torch.Tensor, torch.Tensor],
    ref: tuple[torch.Tensor, torch.Tensor],
    rtol=None,
    atol=None,
) -> dict:
    q_out, k_out = out
    q_ref, k_ref = ref

    if rtol is None:
        rtol = 3e-3 if q_out.dtype in (torch.bfloat16, torch.float16) else 1e-5
    if atol is None:
        atol = 5e-3 if q_out.dtype in (torch.bfloat16, torch.float16) else 1e-5

    diff_q = (q_out - q_ref).abs()
    diff_k = (k_out - k_ref).abs()

    ok_q = torch.allclose(q_out, q_ref, rtol=rtol, atol=atol) and not (
        torch.isnan(q_out).any() or torch.isinf(q_out).any()
    )
    ok_k = torch.allclose(k_out, k_ref, rtol=rtol, atol=atol) and not (
        torch.isnan(k_out).any() or torch.isinf(k_out).any()
    )

    return {
        "ok": bool(ok_q and ok_k),
        "rtol": rtol,
        "atol": atol,
        "absmax_q": float(diff_q.max().item()),
        "absmax_k": float(diff_k.max().item()),
        "mae_q": float(diff_q.mean().item()),
        "mae_k": float(diff_k.mean().item()),
        "mse_q": float(((q_out - q_ref) ** 2).mean().item()),
        "mse_k": float(((k_out - k_ref) ** 2).mean().item()),
        "ref": "rotary_torch",
    }


def workloads(dtype="float32", device="cuda") -> Iterable[dict]:
    # Based on typical transformer configurations
    for batch in [1, 2]:
        for seqlen in [128, 512, 2048]:
            for num_heads in [8, 32]:
                for head_dim in [64, 128]:
                    rotary_dim = head_dim // 2
                    yield {
                        "name": f"{device}_B{batch}_S{seqlen}_H{num_heads}_D{head_dim}_R{rotary_dim}",
                        "batch": batch,
                        "seqlen": seqlen,
                        "num_heads": num_heads,
                        "head_dim": head_dim,
                        "rotary_dim": rotary_dim,
                        "dtype": dtype,
                        "device": device,
                        "seed": 0,
                    }
