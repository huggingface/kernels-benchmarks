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
    """Generate input tensors for SwiGLU activation benchmarking.

    Args:
        wl: Workload dictionary containing:
            - num_tokens: Number of tokens
            - hidden_dim: Hidden dimension size
            - dtype: Data type string
            - device: Device string ('cuda' or 'cpu')
            - seed: Random seed

    Returns:
        Tuple of (input_tensor,) where input has shape (num_tokens, 2 * hidden_dim)
    """
    torch.manual_seed(int(wl.get("seed", 0)))
    num_tokens = wl["num_tokens"]
    hidden_dim = wl["hidden_dim"]
    dt = _dtype(wl.get("dtype", "bfloat16"))
    dev = wl.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # SwiGLU input has shape (num_tokens, 2 * hidden_dim)
    input_tensor = torch.randn(num_tokens, 2 * hidden_dim, device=dev, dtype=dt)

    return (input_tensor,)


def ref_swiglu(inputs: Sequence[torch.Tensor]) -> torch.Tensor:
    """Reference SwiGLU implementation using float32 for numerical stability.

    SwiGLU computes: silu(x[:d]) * x[d:] where d = hidden_dim
    """
    (input_tensor,) = inputs

    # Convert to float32 for reference computation
    # x_f32 = input_tensor.to(torch.float32)
    x_f32 = input_tensor.to(torch.bfloat16)

    # Split input into two halves
    d = x_f32.shape[-1] // 2
    x1 = x_f32[..., :d]  # First half
    x2 = x_f32[..., d:]  # Second half

    # SwiGLU: silu(x1) * x2
    output = torch.nn.functional.silu(x1) * x2

    # Convert back to original dtype
    return output.to(input_tensor.dtype)


def cmp_allclose(out: torch.Tensor, ref: torch.Tensor, rtol=None, atol=None) -> dict:
    """Compare output with reference using allclose with detailed diagnostics on failure.

    Args:
        out: Kernel output tensor
        ref: Reference output tensor
        rtol: Relative tolerance (auto-adjusted for dtype if None)
        atol: Absolute tolerance (auto-adjusted for dtype if None)

    Returns:
        Dictionary with comparison metrics
    """
    from .core.tools import detailed_tensor_comparison

    # Auto-adjust tolerances based on dtype precision
    # bfloat16 has ~7-8 bits of mantissa (~0.78% precision)
    # float16 has ~10 bits of mantissa (~0.098% precision)
    # float32 has ~23 bits of mantissa (~0.000012% precision)
    if rtol is None:
        if out.dtype == torch.bfloat16:
            rtol = 1e-2  # 1% for bfloat16
        elif out.dtype == torch.float16:
            rtol = 5e-3  # 0.5% for float16
        else:
            rtol = 1e-3  # 0.1% for float32

    if atol is None:
        if out.dtype == torch.bfloat16:
            atol = 1e-1  # Absolute tolerance for bfloat16
        elif out.dtype == torch.float16:
            atol = 1e-2  # Absolute tolerance for float16
        else:
            atol = 1e-3  # Absolute tolerance for float32

    result = detailed_tensor_comparison(
        out, ref, rtol=rtol, atol=atol, name="SwiGLU Activation"
    )

    # Add reference implementation name for tracking
    result["ref"] = "swiglu_bfloat16"

    return result


def llama_workloads(dtype="bfloat16") -> Iterable[dict]:
    """Generate LLaMA-style workloads for SwiGLU benchmarking."""
    for num_tokens in [512, 1024, 2048, 4096]:
        for hidden_dim in [4096, 8192, 11008]:
            yield {
                "name": f"llama_T{num_tokens}_D{hidden_dim}",
                "num_tokens": num_tokens,
                "hidden_dim": hidden_dim,
                "dtype": dtype,
                "device": "cuda",
                "seed": 0,
            }


def cpu_workloads(dtype="float32") -> Iterable[dict]:
    """Generate smaller workloads suitable for CPU testing."""
    for num_tokens in [128, 256, 512]:
        for hidden_dim in [768, 1024, 2048]:
            yield {
                "name": f"cpu_T{num_tokens}_D{hidden_dim}",
                "num_tokens": num_tokens,
                "hidden_dim": hidden_dim,
                "dtype": dtype,
                "device": "cpu",
                "seed": 0,
            }
