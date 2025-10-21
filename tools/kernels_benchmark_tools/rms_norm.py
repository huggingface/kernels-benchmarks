from __future__ import annotations
from typing import Sequence, Iterable
import torch


def _dtype(s: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[s]


def gen_inputs(wl: dict) -> Sequence[torch.Tensor]:
    torch.manual_seed(int(wl.get("seed", 0)))
    B, S, D = wl["batch"], wl["seq_len"], wl["hidden_dim"]
    dt = _dtype(wl.get("dtype", "bfloat16"))
    dev = wl.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Generate input tensor
    x = torch.randn(B, S, D, device=dev, dtype=dt) * 0.5

    # Generate weight tensor
    weight = torch.ones(D, device=dev, dtype=dt)

    return (x, weight)


def ref_rms_norm(inputs: Sequence[torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    """Reference RMS norm implementation using float32 for numerical stability."""
    x, weight = inputs

    # Convert to float32 for reference computation
    x_f32 = x.to(torch.float32)
    weight_f32 = weight.to(torch.float32)

    # Compute RMS norm in float32
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_normalized = x_f32 * torch.rsqrt(variance + eps)
    output = x_normalized * weight_f32

    # Convert back to original dtype
    return output.to(x.dtype)


def cmp_allclose(out: torch.Tensor, ref: torch.Tensor, rtol=1e-3, atol=1e-3) -> dict:
    """Compare output with reference using allclose."""
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
        "ref": "rms_norm_fp32",
    }


def llama_workloads(dtype="bfloat16") -> Iterable[dict]:
    """Generate LLaMA-style workloads for RMS norm benchmarking."""
    for seq_len in [512, 1024, 2048, 4096]:
        for hidden_dim in [4096, 8192]:
            yield {
                "name": f"llama_S{seq_len}_D{hidden_dim}",
                "batch": 1,
                "seq_len": seq_len,
                "hidden_dim": hidden_dim,
                "dtype": dtype,
                "device": "cuda",
                "seed": 0,
            }


def cpu_workloads(dtype="float32") -> Iterable[dict]:
    """Generate smaller workloads suitable for CPU testing."""
    for seq_len in [128, 256, 512]:
        for hidden_dim in [768, 1024]:
            yield {
                "name": f"cpu_S{seq_len}_D{hidden_dim}",
                "batch": 1,
                "seq_len": seq_len,
                "hidden_dim": hidden_dim,
                "dtype": dtype,
                "device": "cpu",
                "seed": 0,
            }
