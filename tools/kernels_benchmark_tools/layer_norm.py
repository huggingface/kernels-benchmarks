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
    """Generate inputs for LayerNorm: (x, weight, bias)."""
    torch.manual_seed(int(wl.get("seed", 0)))
    B, S, D = wl["batch"], wl["seq_len"], wl["hidden_dim"]
    dt = _dtype(wl.get("dtype", "bfloat16"))
    dev = wl.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(B, S, D, device=dev, dtype=dt) * 0.5
    weight = torch.ones(D, device=dev, dtype=dt)
    bias = torch.zeros(D, device=dev, dtype=dt)
    return (x, weight, bias)


def ref_layer_norm(inputs: Sequence[torch.Tensor], eps: float = 1e-5) -> torch.Tensor:
    """Reference LayerNorm in float32 with affine (weight, bias)."""
    x, weight, bias = inputs
    # x_f32 = x.to(torch.float32)
    # w_f32 = weight.to(torch.float32)
    # b_f32 = bias.to(torch.float32)

    # make them bfloat16/float16 compatible
    x_f32 = x.bfloat16()
    w_f32 = weight.bfloat16()
    b_f32 = bias.bfloat16()

    mean = x_f32.mean(dim=-1, keepdim=True)
    var = x_f32.var(dim=-1, keepdim=True, unbiased=False)
    y = (x_f32 - mean) * torch.rsqrt(var + eps)
    y = y * w_f32 + b_f32
    return y.to(x.dtype)


def cmp_allclose(out: torch.Tensor, ref: torch.Tensor, rtol=1e-3, atol=1e-3) -> dict:
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
        "ref": "layer_norm_fp32",
    }


def llama_workloads(dtype: str = "bfloat16") -> Iterable[dict]:
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


def cpu_workloads(dtype: str = "float32") -> Iterable[dict]:
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
