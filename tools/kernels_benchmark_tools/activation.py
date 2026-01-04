from __future__ import annotations
from typing import Sequence, Iterable
import torch


def gen_inputs(wl: dict) -> Sequence[torch.Tensor]:
    torch.manual_seed(int(wl.get("seed", 0)))
    num_tokens = wl["num_tokens"]
    hidden_dim = wl["hidden_dim"]
    dt = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(wl.get("dtype", "bfloat16"))
    dev = wl.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # SwiGLU input has shape (num_tokens, 2 * hidden_dim)
    input_tensor = torch.randn(num_tokens, 2 * hidden_dim, device=dev, dtype=dt)

    return (input_tensor,)


def ref_impl(inputs):
    (input_tensor,) = inputs

    d = input_tensor.shape[-1] // 2
    x1 = input_tensor[..., :d]
    x2 = input_tensor[..., d:]

    return torch.nn.functional.silu(x1) * x2


def cmp_allclose(out: torch.Tensor, ref: torch.Tensor, rtol=None, atol=None) -> dict:
    if rtol is None:
        rtol = 2e-2 if out.dtype in (torch.bfloat16, torch.float16) else 1e-3
    if atol is None:
        atol = 2e-2 if out.dtype in (torch.bfloat16, torch.float16) else 1e-3

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
        "ref": "swiglu_fp32",
    }


def workloads(dtype="float32", device="cuda") -> Iterable[dict]:
    for num_tokens in [128, 256, 512]:
        for hidden_dim in [768, 1024, 2048]:
            yield {
                "name": f"{device}_T{num_tokens}_D{hidden_dim}",
                "num_tokens": num_tokens,
                "hidden_dim": hidden_dim,
                "dtype": dtype,
                "device": device,
                "seed": 0,
            }
