from __future__ import annotations
from typing import Sequence, Iterable
import torch
import torch.nn.functional as F


def gen_inputs(wl: dict) -> Sequence[torch.Tensor]:
    torch.manual_seed(int(wl.get("seed", 0)))
    batch = wl["batch"]
    dim = wl["dim"]
    seqlen = wl["seqlen"]
    width = wl.get("width", 4)
    dt = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(wl.get("dtype", "bfloat16"))
    dev = wl.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Causal conv1d input shape: (batch, dim, seqlen)
    input_tensor = torch.randn(batch, dim, seqlen, device=dev, dtype=dt)
    # Weight shape: (dim, width), dtype is float32 as per the kernel requirements
    weight = torch.randn(dim, width, device=dev, dtype=torch.float32)
    # Bias shape: (dim,), dtype is float32
    bias = torch.randn(dim, device=dev, dtype=torch.float32)

    return (input_tensor, weight, bias)


def ref_impl(inputs: Sequence[torch.Tensor]) -> torch.Tensor:
    input_tensor, weight, bias = inputs

    # Reference implementation using PyTorch F.conv1d
    # Convert to float32 for computation
    x_fp32 = input_tensor.to(weight.dtype)
    dim = weight.shape[0]
    width = weight.shape[1]
    seqlen = input_tensor.shape[-1]

    # Perform causal conv1d with appropriate padding
    # groups=dim makes it a depthwise convolution
    out = F.conv1d(x_fp32, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)

    # Truncate to original sequence length (causal convolution)
    out = out[..., :seqlen]

    # Convert back to original dtype
    return out.to(input_tensor.dtype)


def cmp_allclose(out: torch.Tensor, ref: torch.Tensor, rtol=None, atol=None) -> dict:
    if rtol is None:
        rtol = 3e-3 if out.dtype in (torch.bfloat16, torch.float16) else 1e-3
    if atol is None:
        atol = 5e-3 if out.dtype in (torch.bfloat16, torch.float16) else 1e-3

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
        "ref": "causal_conv1d_fp32",
    }


def workloads(dtype="float32", device="cuda") -> Iterable[dict]:
    # Based on the test patterns in the causal-conv1d tests
    for batch in [2, 4]:
        for dim in [64, 2048]:
            for seqlen in [128, 512, 2048]:
                for width in [2, 4]:
                    yield {
                        "name": f"{device}_B{batch}_D{dim}_S{seqlen}_W{width}",
                        "batch": batch,
                        "dim": dim,
                        "seqlen": seqlen,
                        "width": width,
                        "dtype": dtype,
                        "device": device,
                        "seed": 0,
                    }
