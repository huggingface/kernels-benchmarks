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

    x = torch.randn(B, S, D, device=dev, dtype=dt) * 0.5
    weight = torch.ones(D, device=dev, dtype=dt)
    bias = torch.zeros(D, device=dev, dtype=dt)
    return (x, weight, bias)


def ref_impl(inputs: Sequence[torch.Tensor], eps: float = 1e-5) -> torch.Tensor:
    x, weight, bias = inputs
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    y = (x - mean) * torch.rsqrt(var + eps)
    y = y * weight + bias
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


def workloads(dtype: str = "bfloat16", device: str = "cpu") -> Iterable[dict]:
    for batch in [1, 4, 16]:
        for seq_len in [128, 512, 1024, 2048]:
            for hidden_dim in [1024, 2048, 4096, 8192]:
                yield {
                    "name": f"LN_B{batch}_S{seq_len}_D{hidden_dim}",
                    "batch": batch,
                    "seq_len": seq_len,
                    "hidden_dim": hidden_dim,
                    "dtype": dtype,
                    "device": device,
                    "seed": 0,
                }
