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


# Returns the spacing (1 ULP) between adjacent bfloat16 values at each x.
# Used to verify numerical correctness by checking results differ only
# within one representable rounding step of bfloat16 precision.
def bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    xb = x.to(torch.bfloat16)
    next_val = torch.nextafter(xb, torch.full_like(xb, float("inf")))
    return (next_val - xb).abs().to(torch.float32)


def cmp_allclose(out: torch.Tensor, ref: torch.Tensor, rtol=1e-3, atol=1e-2) -> dict:
    diff = (out - ref).abs()
    ulp = bf16_ulp(torch.maximum(out.abs(), ref.abs())).max().item()
    if atol < ulp:
        atol = ulp  # ensure at least 1 ULP tolerance
    ok = torch.allclose(out, ref, rtol=rtol, atol=ulp) and not (
        torch.isnan(out).any() or torch.isinf(out).any()
    )
    return {
        "ok": bool(ok),
        "rtol": rtol,
        "atol": atol,
        "absmax": float(diff.max().item()),
        "mae": float(diff.mean().item()),
        "mse": float(((out - ref) ** 2).mean().item()),
        "ref": "layer_norm_ref",
    }


def workloads(dtype: str = "bfloat16", device: str = "cpu") -> Iterable[dict]:
    for batch in [16]:
        for seq_len in [2048, 4096]:
            for hidden_dim in [4096, 8_192]:
                yield {
                    "name": f"LN_B{batch}_S{seq_len}_D{hidden_dim}",
                    "batch": batch,
                    "seq_len": seq_len,
                    "hidden_dim": hidden_dim,
                    "dtype": dtype,
                    "device": device,
                    "seed": 0,
                }
