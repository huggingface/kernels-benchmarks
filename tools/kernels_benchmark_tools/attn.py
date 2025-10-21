from __future__ import annotations
from typing import Sequence, Iterable
import torch


def _dtype(s: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[s]


def gen_qkv(wl: dict) -> Sequence[torch.Tensor]:
    torch.manual_seed(int(wl.get("seed", 0)))
    B, S, H, D = wl["batch"], wl["seq_len"], wl["heads"], wl["head_dim"]
    dt = _dtype(wl.get("dtype", "bfloat16"))
    dev = wl.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    shape = (B, S, H, D)

    def sample():
        base = torch.randn(shape, device=dev, dtype=dt)
        mult = torch.randint(1, 5, shape, device=dev, dtype=dt)
        return base * mult

    return (sample(), sample(), sample())


def ref_math(inputs: Sequence[torch.Tensor]) -> torch.Tensor:
    q, k, v = inputs
    qf, kf, vf = (x.to(torch.float32).transpose(1, 2).contiguous() for x in (q, k, v))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        o = torch.nn.functional.scaled_dot_product_attention(qf, kf, vf)
    return o.transpose(1, 2).contiguous().to(q.dtype)


def cmp_allclose(out: torch.Tensor, ref: torch.Tensor, rtol=2e-2, atol=2e-2) -> dict:
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
        "ref": "sdpa_math_fp32",
    }


def flux_workloads(dtype="bfloat16") -> Iterable[dict]:
    base = 4096
    for L in [128, 256, 320, 384, 448, 512]:
        yield {
            "name": f"flux_L{L}",
            "batch": 1,
            "seq_len": base + L,
            "heads": 24,
            "head_dim": 128,
            "dtype": dtype,
            "device": "cuda",
            "seed": 0,
        }
