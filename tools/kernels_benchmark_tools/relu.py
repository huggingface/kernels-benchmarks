from __future__ import annotations
from typing import Sequence, Iterable
import torch
import torch.nn.functional as F


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

    # ReLU input: simple tensor with shape (num_tokens, hidden_dim)
    input_tensor = torch.randn(num_tokens, hidden_dim, device=dev, dtype=dt)

    return (input_tensor,)


def ref_impl(inputs: Sequence[torch.Tensor]) -> torch.Tensor:
    (input_tensor,) = inputs

    # Reference implementation using PyTorch F.relu
    return F.relu(input_tensor)


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
        "ref": "relu_torch",
    }


def workloads(dtype="float32", device="cuda") -> Iterable[dict]:
    for num_tokens in [128, 512, 1024, 2048]:
        for hidden_dim in [768, 1024, 2048, 4096]:
            yield {
                "name": f"{device}_T{num_tokens}_D{hidden_dim}",
                "num_tokens": num_tokens,
                "hidden_dim": hidden_dim,
                "dtype": dtype,
                "device": device,
                "seed": 0,
            }
