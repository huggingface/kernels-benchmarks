# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "torch==2.8.0",
#     "kernels-benchmark-tools",
#     "kernels",
# ]
#
# [tool.uv.sources]
# kernels-benchmark-tools = { path = "../../../tools", editable = true }
# ///
import torch
# kernels-benchmark-tools = { path = "../../../../../tools", editable = true }
import sys
from kernels_benchmark_tools import KernelTypeEnum, run_benchmark
from kernels import get_kernel

# Load the rotary kernel
rotary = get_kernel("kernels-community/rotary")


def hf_kernels_rotary(query, key, cos, sin, conj=False):
    rotary_dim = cos.shape[-1]

    # Clone to avoid modifying inputs
    q_out = query.clone()
    k_out = key.clone()

    # Apply rotation to query
    q1 = q_out[..., :rotary_dim]
    q2 = q_out[..., rotary_dim : 2 * rotary_dim]
    rotary.apply_rotary(q1, q2, cos, sin, q1, q2, conj)

    # Apply rotation to key
    k1 = k_out[..., :rotary_dim]
    k2 = k_out[..., rotary_dim : 2 * rotary_dim]
    rotary.apply_rotary(k1, k2, cos, sin, k1, k2, conj)

    # print("HF Kernels q_out:", q_out.sum().item())
    # print("HF Kernels k_out:", k_out.sum().item())

    # # first 5 elements for quick verification
    # print("HF Kernels q_out (first 5):", q_out.flatten()[:5])
    # print("HF Kernels k_out (first 5):", k_out.flatten()[:5])

    return q_out, k_out


run_benchmark(
    kernel_type=KernelTypeEnum.ROTARY,
    impl_name="hf_kernels_rotary",
    impl_tags={"family": "hf-kernels", "backend": "cuda"},
    impl_func=hf_kernels_rotary,
    #
    reps=1,
    warmup=0,
    dtype="float32",
)