on_github: huggingface/kernels-uvnotes
---

# HF Kernels LayerNorm Implementation

Based on kernels-community `layer-norm` kernel.

## LayerNorm Benchmark (HF Kernels)

```python id=benchmark outputs=ln.jsonl
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "torch==2.8.0",
#     "kernels",
#     "kernels-benchmark-tools",
# ]
#
# [tool.uv.sources]
# kernels-benchmark-tools = { path = "../../../../../tools", editable = true }
# ///
import torch
from kernels import get_kernel
import kernels_benchmark_tools as kbt

layer_norm_kernel = get_kernel("kernels-community/layer-norm")

def hf_kernels_layer_norm(x, weight, bias, eps: float = 1e-5):
    B, S, D = x.shape
    # The kernel expects [N, D] input; support beta (bias) if provided.
    out = layer_norm_kernel.dropout_add_ln_fwd(
        input=x.view(-1, D),
        gamma=weight,
        beta=bias,
        rowscale=None,
        colscale=None,
        x0_subset=None,
        z_subset=None,
        dropout_p=0.0,
        epsilon=eps,
        rowscale_const=1.0,
        z_numrows=S,
        gen=None,
        residual_in_fp32=False,
        is_rms_norm=False,
    )[0].view(B, S, D)
    return out

kbt.add(
    "hf_kernels_layer_norm",
    hf_kernels_layer_norm,
    tags={"family": "hf-kernels", "repo": "kernels-community/layer-norm", "op": "layer_norm"},
)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "float32" if device == "cpu" else "bfloat16"

    wl = list(kbt.layer_norm.llama_workloads(dtype)) if device == "cuda" else list(kbt.layer_norm.cpu_workloads(dtype))

    kbt.run(
        wl,
        jsonl="ln.jsonl",
        reps=5,
        warmup=2,
        gen=kbt.layer_norm.gen_inputs,
        ref=kbt.layer_norm.ref_layer_norm,
        cmp=kbt.layer_norm.cmp_allclose,
        profile_trace=False,
    )
    kbt.summarize(["ln.jsonl"])
```
