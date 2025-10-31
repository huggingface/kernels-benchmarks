from . import attn
from . import rms_norm
from . import layer_norm
from . import openai_moe
from . import activation
from . import causal_conv1d
from . import relu
from . import rotary
from . import deformable_detr
from . import core
from .core.harness import add, run
from .core.tools import summarize, viz
from enum import Enum
import torch

# NOTE: KernelTypeEnum and run_benchmark are the main entry points for benchmarking kernels.
# When adding a new kernel we must add a new type here and implement the required functions
# in a new module located in this directory.


# Kernels benchmarking types
class KernelTypeEnum(str, Enum):
    ATTENTION = "attention"
    RMS_NORM = "rms_norm"
    LAYER_NORM = "layer_norm"
    OPENAI_MOE = "openai_moe"
    ACTIVATION = "activation"
    CAUSAL_CONV1D = "causal_conv1d"
    RELU = "relu"
    ROTARY = "rotary"
    DEFORMABLE_DETR = "deformable_detr"


# Map from type to module for implementation
KERNEL_MODULES = {
    KernelTypeEnum.ATTENTION: attn,
    KernelTypeEnum.RMS_NORM: rms_norm,
    KernelTypeEnum.LAYER_NORM: layer_norm,
    KernelTypeEnum.OPENAI_MOE: openai_moe,
    KernelTypeEnum.ACTIVATION: activation,
    KernelTypeEnum.CAUSAL_CONV1D: causal_conv1d,
    KernelTypeEnum.RELU: relu,
    KernelTypeEnum.ROTARY: rotary,
    KernelTypeEnum.DEFORMABLE_DETR: deformable_detr,
}

## Benchmarking Functions


# Run the benchmark for a given kernel type
def run_benchmark(
    kernel_type: KernelTypeEnum,
    impl_name: str | None = None,
    impl_tags: dict | None = None,
    impl_func=None,
    reps: int = 5,
    warmup: int = 2,
    dtype: str | None = None,
    device: str | None = None,
    **kwargs,
):
    # Determine device and dtype (TODO: allow user override)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or ("float32" if device == "cpu" else "bfloat16")

    # Get the kernel module based on type (TODO: handle invalid type)
    kernel_module = KERNEL_MODULES[kernel_type]

    # Register the implementation
    add(
        impl_name,
        impl_func,
        tags=impl_tags or {},
    )

    # Generate workloads
    wl = list(kernel_module.workloads(dtype=dtype, device=device))

    print(
        f"Running {kernel_type.value} benchmark on {device} with {len(wl)} workloads."
    )

    # Run the benchmark given the generated workloads and kernel module functions
    run(
        wl,
        jsonl=f"{kernel_type.value}.jsonl",
        reps=reps,
        warmup=warmup,
        gen=kernel_module.gen_inputs,
        ref=kernel_module.ref_impl,
        cmp=kernel_module.cmp_allclose,
        profile_trace=True,
        **kwargs,
    )

    # Print a summary of the results for debugging/inspection
    summarize([f"{kernel_type.value}.jsonl"])


__all__ = [
    "attn",
    "rms_norm",
    "layer_norm",
    "openai_moe",
    "activation",
    "causal_conv1d",
    "relu",
    "rotary",
    "deformable_detr",
    "core",
    "add",
    "run",
    "summarize",
    "viz",
    "KernelTypeEnum",
]
