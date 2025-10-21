# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "torch",
#     "kernels-benchmark-tools",
# ]
#
# [tool.uv.sources]
# kernels-benchmark-tools = { path = "/home/ubuntu/Projects/kernels-benchmarks-consolidated/tools", editable = true }
# ///
import torch
import sys
import os
import kernels_benchmark_tools as kbt


def torch_rms_norm(x, weight, eps=1e-6):
    """Standard PyTorch RMS norm implementation."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normalized = x * torch.rsqrt(variance + eps)
    return x_normalized * weight


def torch_rms_norm_fp32(x, weight, eps=1e-6):
    """RMS norm with float32 computation for higher precision."""
    x_f32 = x.to(torch.float32)
    weight_f32 = weight.to(torch.float32)

    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_normalized = x_f32 * torch.rsqrt(variance + eps)
    output = x_normalized * weight_f32

    return output.to(x.dtype)


def torch_layer_norm_as_rms(x, weight, eps=1e-6):
    """Use LayerNorm without bias as RMS norm approximation."""
    # LayerNorm without bias is similar to RMS norm but with mean centering
    return torch.nn.functional.layer_norm(
        x, x.shape[-1:], weight=weight, bias=None, eps=eps
    )


def manual_rms_norm(x, weight, eps=1e-6):
    """Manual implementation with explicit operations."""
    # Compute variance manually
    squared = x * x
    variance = torch.mean(squared, dim=-1, keepdim=True)

    # Compute normalization factor
    inv_std = torch.rsqrt(variance + eps)

    # Apply normalization and scaling
    normalized = x * inv_std
    return normalized * weight


kbt.add(
    "torch_rms_norm",
    torch_rms_norm,
    tags={"family": "torch", "precision": "native", "compile": "none"},
)
kbt.add(
    "torch_rms_norm_fp32",
    torch_rms_norm_fp32,
    tags={"family": "torch", "precision": "fp32", "compile": "none"},
)
kbt.add(
    "torch_layer_norm_as_rms",
    torch_layer_norm_as_rms,
    tags={"family": "torch", "type": "layer_norm", "compile": "none"},
)
kbt.add(
    "manual_rms_norm",
    manual_rms_norm,
    tags={"family": "manual", "precision": "native", "compile": "none"},
)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "float32" if device == "cpu" else "bfloat16"

    # Check for profiling flag
    enable_profiling = "--profile" in sys.argv

    # Generate workloads based on device
    if device == "cuda":
        wl = list(kbt.rms_norm.llama_workloads(dtype=dtype))
        # Take a subset for faster testing
        wl = wl[:4]  # First 4 workloads
    else:
        wl = list(kbt.rms_norm.cpu_workloads(dtype=dtype))
        # Take a subset for faster testing
        wl = wl[:3]  # First 3 workloads

    print(f"Running RMS norm benchmarks on {device} with {dtype}")
    print(f"Testing {len(wl)} workloads")
    if enable_profiling:
        print("Profiling enabled - will print trace summaries\n")

    kbt.run(
        wl,
        jsonl="rms_norm.jsonl",
        reps=5,
        warmup=2,
        gen=kbt.rms_norm.gen_inputs,
        ref=kbt.rms_norm.ref_rms_norm,
        cmp=kbt.rms_norm.cmp_allclose,
        profile_trace=enable_profiling,  # Enable with --profile flag
    )

    # Generate summary and visualization
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    kbt.summarize(["rms_norm.jsonl"])

    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATION")
    print("=" * 60)

    try:
        kbt.viz(["rms_norm.jsonl"])
        print("Visualization saved as latency.png")
    except ImportError:
        print("Visualization requires matplotlib. Install with: uv add matplotlib")
    except Exception as e:
        print(f"Visualization failed: {e}")
