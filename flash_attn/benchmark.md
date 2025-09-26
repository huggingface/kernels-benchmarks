---
title: "Flash Attention Benchmark"
author: "uvnote"
theme: "dark"
syntax_theme: "monokai"
show_line_numbers: true
collapse_code: false
custom_css: |
    #output-setup {
        overflow-x: auto;
    }
    .cell-output {
        overflow: scroll;
    }
    .cell-stdout {
        width: max-content;
        overflow: scroll;
    }
    .cell-stderr {
        width: max-content;
        overflow: scroll;
        max-height: 300px;
    }
---


```python id=nv
import subprocess

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
```


```python id=benchmark
# /// script
# dependencies = [
#   "numpy",
#   "torch",
#   "kernels",
#   "pandas",
#   "matplotlib"
# ]
# ///
# Benchmarking common shapes for Flux 1024x1024px image + varying text sequence lengths

import functools
import os
import pathlib

import matplotlib.pyplot as plt
import torch
import torch._dynamo.config
import triton
import triton.language as tl

try:
    from flash_attn import flash_attn_func
except:
    flash_attn_func = None
    print("Flash Attention 2 not found.")

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
except:
    flash_attn_3_func = None
    print("Flash Attention 3 not found.")

try:
    from kernels import get_kernel
    hf_kernels_flash_attn = get_kernel("kernels-community/flash-attn")
    hf_kernels_flash_attn_3 = get_kernel("kernels-community/flash-attn3")
except:
    hf_kernels_flash_attn = None
    hf_kernels_flash_attn_3 = None
    print("HF Kernels not found.")

try:
    from sageattention import sageattn_qk_int8_pv_fp16_cuda, sageattn_qk_int8_pv_fp16_triton, sageattn_qk_int8_pv_fp8_cuda_sm90
except:
    sageattn_qk_int8_pv_fp16_cuda = None
    sageattn_qk_int8_pv_fp16_triton = None
    sageattn_qk_int8_pv_fp8_cuda_sm90 = None
    print("SageAttention not found.")

try:
    from transformer_engine.pytorch.attention import DotProductAttention
except:
    DotProductAttention = None
    print("Transformer Engine not found.")

try:
    import xformers.ops as xops
except:
    xops = None
    print("xFormers not found.")


plt.rcParams.update({
    "figure.figsize": (12, 10),
    "figure.dpi": 120,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.loc": "best",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# We want to compare the best compiled version for each specific shape (dynamic=False)
torch._dynamo.config.cache_size_limit = 10000

# We need to suppress_errors for FA3 to work. It makes it run in eager mode.
# I can't seem to get it to work any other way under torch.compile, so any suggestions are welcome!
torch._dynamo.config.suppress_errors = True

# output_dir = pathlib.Path("dump_attention_benchmark")
# output_dir.mkdir(parents=True, exist_ok=True)

output_dir = pathlib.Path(".") # output to current directory for upload

batch_size = 1
num_attention_heads = 24
attention_head_dim = 128
image_sequence_length = 4096  # 1024x1024px
text_sequence_lengths = [128, 256, 320, 384, 448, 512]
sequence_lengths = [image_sequence_length + i for i in text_sequence_lengths]


def _attention_torch(query, key, value, *, backend):
    query, key, value = (x.transpose(1, 2).contiguous() for x in (query, key, value))
    with torch.nn.attention.sdpa_kernel(backend):
        out = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    out = out.transpose(1, 2).contiguous()
    return out


_compiled_attention_torch_default = torch.compile(_attention_torch, mode="default", fullgraph=True, dynamic=False)
def _attention_torch_compile_default(query, key, value, *, backend):
    return _compiled_attention_torch_default(query, key, value, backend=backend)


_compiled_attention_torch_max_autotune = torch.compile(_attention_torch, mode="max-autotune", fullgraph=True, dynamic=False)
def _attention_torch_compile_max_autotune(query, key, value, *, backend):
    return _compiled_attention_torch_max_autotune(query, key, value, backend=backend)


def _attention_flash_attn_2(query, key, value):
    return flash_attn_func(query, key, value)


_compiled_flash_attn_2_default = torch.compile(_attention_flash_attn_2, mode="default", fullgraph=True, dynamic=False)
def _attention_flash_attn_2_compile_default(query, key, value):
    return _compiled_flash_attn_2_default(query, key, value)


_compiled_flash_attn_2_max_autotune = torch.compile(_attention_flash_attn_2, mode="max-autotune", fullgraph=True, dynamic=False)
def _attention_flash_attn_2_compile_max_autotune(query, key, value):
    return _compiled_flash_attn_2_max_autotune(query, key, value)


# For fullgraph=True tracing to be compatible
@torch.library.custom_op("flash_attn_3::_flash_attn_forward", mutates_args=(), device_types="cuda")
def _wrapped_flash_attn_3(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    out, lse = flash_attn_3_func(query, key, value)
    return out


@torch.library.register_fake("flash_attn_3::_flash_attn_forward")
def _(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(query)


def _attention_flash_attn_3(query, key, value):
    out = _wrapped_flash_attn_3(query, key, value)
    return out


_compiled_flash_attn_3_default = torch.compile(_attention_flash_attn_3, mode="default", fullgraph=True, dynamic=False)
def _attention_flash_attn_3_compile_default(query, key, value):
    return _compiled_flash_attn_3_default(query, key, value)


_compiled_flash_attn_3_max_autotune = torch.compile(_attention_flash_attn_3, mode="max-autotune", fullgraph=True, dynamic=False)
def _attention_flash_attn_3_compile_max_autotune(query, key, value):
    return _compiled_flash_attn_3_max_autotune(query, key, value)


def _attention_hf_kernels_flash_attn(query, key, value):
    return hf_kernels_flash_attn.fwd(query, key, value, is_causal=False)[0]


def _attention_hf_kernels_flash_attn3(query, key, value):
    return hf_kernels_flash_attn_3.flash_attn_func(query, key, value, causal=False)[0]


def _attention_sageattn_qk_int8_pv_fp16_cuda(query, key, value):
    return sageattn_qk_int8_pv_fp16_cuda(query, key, value, tensor_layout="NHD")


def _attention_sageattn_qk_int8_pv_fp16_triton(query, key, value):
    return sageattn_qk_int8_pv_fp16_triton(query, key, value, tensor_layout="NHD")


def _attention_sageattn_qk_int8_pv_fp8_cuda_sm90(query, key, value):
    return sageattn_qk_int8_pv_fp8_cuda_sm90(query, key, value, tensor_layout="NHD")


if DotProductAttention is not None:
    def set_te_backend(backend):
        # must be applied before first use of
        # transformer_engine.pytorch.attention
        os.environ["NVTE_FLASH_ATTN"] = '0'
        os.environ["NVTE_FUSED_ATTN"] = '0'
        os.environ["NVTE_UNFUSED_ATTN"] = '0'
        if backend == 'flash':
            os.environ["NVTE_FLASH_ATTN"] = '1'
        if backend == 'fused':
            os.environ["NVTE_FUSED_ATTN"] = '1'
        if backend == 'unfused':
            os.environ["NVTE_UNFUSED_ATTN"] = '1'
    
    set_te_backend("fused")
    te_attn_fn = DotProductAttention(
        num_attention_heads=num_attention_heads,
        kv_channels=attention_head_dim,
        qkv_format="bshd",
        attn_mask_type="no_mask",
    )
else:
    def te_attn_fn(query, key, value):
        raise RuntimeError("Transformer Engine is not available. Please install it for TE-based attention.")

def _attention_te(query, key, value):
    out = te_attn_fn(query, key, value)
    out = out.unflatten(2, (num_attention_heads, attention_head_dim))
    return out


# Cannot fullgraph compile TE
_compiled_te_attn_fn_default = torch.compile(_attention_te, mode="default", fullgraph=False, dynamic=False)
def _attention_te_compile_default(query, key, value):
    return _compiled_te_attn_fn_default(query, key, value)


# Cannot fullgraph compile TE
_compiled_te_attn_fn_max_autotune = torch.compile(_attention_te, mode="max-autotune", fullgraph=False, dynamic=False)
def _attention_te_compile_max_autotune(query, key, value):
    return _compiled_te_attn_fn_max_autotune(query, key, value)


def _attention_xformers(query, key, value):
    return xops.memory_efficient_attention(query, key, value)


_compiled_xformers_default = torch.compile(_attention_xformers, mode="default", fullgraph=True, dynamic=False)
def _attention_xformers_compile_default(query, key, value):
    return _compiled_xformers_default(query, key, value)


_compiled_xformers_max_autotune = torch.compile(_attention_xformers, mode="max-autotune", fullgraph=True, dynamic=False)
def _attention_xformers_compile_max_autotune(query, key, value):
    return _compiled_xformers_max_autotune(query, key, value)


attention_ops = {}
attention_ops["torch_cudnn"] = functools.partial(_attention_torch, backend=torch.nn.attention.SDPBackend.CUDNN_ATTENTION)
attention_ops["torch_cudnn_compile_d"] = functools.partial(_attention_torch_compile_default, backend=torch.nn.attention.SDPBackend.CUDNN_ATTENTION)
attention_ops["torch_cudnn_compile_ma"] = functools.partial(_attention_torch_compile_max_autotune, backend=torch.nn.attention.SDPBackend.CUDNN_ATTENTION)
attention_ops["torch_flash"] = functools.partial(_attention_torch, backend=torch.nn.attention.SDPBackend.FLASH_ATTENTION)
attention_ops["torch_flash_compile_d"] = functools.partial(_attention_torch_compile_default, backend=torch.nn.attention.SDPBackend.FLASH_ATTENTION)
attention_ops["torch_flash_compile_ma"] = functools.partial(_attention_torch_compile_max_autotune, backend=torch.nn.attention.SDPBackend.FLASH_ATTENTION)
if hf_kernels_flash_attn is not None:
    attention_ops["hf_flash_attn"] = _attention_hf_kernels_flash_attn
    attention_ops["hf_flash_attn3"] = _attention_hf_kernels_flash_attn3
if flash_attn_func is not None:
    attention_ops["flash_attn_2"] = _attention_flash_attn_2
    attention_ops["flash_attn_2_compile_d"] = _attention_flash_attn_2_compile_default
    attention_ops["flash_attn_2_compile_ma"] = _attention_flash_attn_2_compile_max_autotune
if flash_attn_3_func is not None:
    attention_ops["flash_attn_3"] = _attention_flash_attn_3
    attention_ops["flash_attn_3_compile_d"] = _attention_flash_attn_3_compile_default
    attention_ops["flash_attn_3_compile_ma"] = _attention_flash_attn_3_compile_max_autotune
if sageattn_qk_int8_pv_fp16_cuda is not None:
    attention_ops["sageattn_qk_int8_pv_fp16_cuda"] = _attention_sageattn_qk_int8_pv_fp16_cuda
    attention_ops["sageattn_qk_int8_pv_fp16_triton"] = _attention_sageattn_qk_int8_pv_fp16_triton
    if torch.cuda.get_device_capability()[0] >= 9:
        attention_ops["sageattn_qk_int8_pv_fp8_cuda_sm90"] = _attention_sageattn_qk_int8_pv_fp8_cuda_sm90
if DotProductAttention is not None:
    attention_ops["te_fused"] = _attention_te
    attention_ops["te_fused_compile_d"] = _attention_te_compile_default
    attention_ops["te_fused_compile_ma"] = _attention_te_compile_max_autotune
if xops is not None:
    attention_ops["xformers"] = _attention_xformers
    attention_ops["xformers_compile_d"] = _attention_xformers_compile_default
    attention_ops["xformers_compile_ma"] = _attention_xformers_compile_max_autotune


def get_color_and_linestyle(n: int) -> tuple[str, str]:
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999"]
    line_styles = ["-", ":", "-.", "--"]
    if n > len(colors) * len(line_styles):
        raise ValueError(f"Required {n=} styles but maximum is {len(colors) * len(line_styles)}")
    styles = []
    for i in range(n):
        color = colors[i % len(colors)]
        linestyle = line_styles[i // len(colors)]
        styles.append((color, linestyle))
    return styles


def correctness():
    for seq_len in sequence_lengths:
        shape = (batch_size, seq_len, num_attention_heads, attention_head_dim)
        print(f"\n\n===== Testing shape: {shape} =====")
        
        query = torch.randn(shape, device="cuda", dtype=torch.float32)
        key = torch.randn(shape, device="cuda", dtype=torch.float32)
        value = torch.randn(shape, device="cuda", dtype=torch.float32)

        golden_truth = _attention_torch(query, key, value, backend=torch.nn.attention.SDPBackend.MATH)
        query, key, value = (x.bfloat16() for x in (query, key, value))

        for name, fn in attention_ops.items():
            out = fn(query, key, value)
            absdiff = (out - golden_truth).abs()
            absmax = torch.max(absdiff)
            mae = torch.mean(absdiff)
            mse = torch.mean((golden_truth - out) ** 2)
            print(f"{name:<30}: absmax={absmax:.6f}, mae={mae:.6f}, mse={mse:.6f}")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=sequence_lengths,
        x_log=False,
        line_arg="provider",
        line_vals=list(attention_ops.keys()),
        line_names=[x.removeprefix("solution_") for x in attention_ops.keys()],
        ylabel="Time (ms)",
        styles=get_color_and_linestyle(len(attention_ops)),
        plot_name="Attention Benchmark",
        args={},
    )
)
def benchmark_fn(seq_len: int, provider: str):
    torch.manual_seed(0)
    
    shape = (batch_size, seq_len, num_attention_heads, attention_head_dim)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * torch.randint(1, 5, shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * torch.randint(1, 5, shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * torch.randint(1, 5, shape, device="cuda", dtype=torch.bfloat16)
    
    fn = attention_ops[provider]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: fn(query, key, value),
        warmup=3,
        rep=10,
        quantiles=[0.5, 0.2, 0.8],
    )
    return ms, max_ms, min_ms


with torch.inference_mode():
    correctness()
    fig = benchmark_fn.run(print_data=True, save_path=output_dir.as_posix())

```