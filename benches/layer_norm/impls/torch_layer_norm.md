on_github: huggingface/kernels-uvnotes
---

# Torch LayerNorm Implementation

## GPU Info

```python id=nv
import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
```

## LayerNorm Benchmark (PyTorch)

```python id=benchmark outputs=ln.jsonl
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "torch==2.8.0",
#     "kernels-benchmark-tools",
# ]
#
# [tool.uv.sources]
# kernels-benchmark-tools = { path = "../../../../../tools", editable = true }
# ///
import torch
import kernels_benchmark_tools as kbt


def torch_layer_norm(x, weight, bias, eps: float = 1e-5):
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)

kbt.add(
    "torch_layer_norm",
    torch_layer_norm,
    tags={"family": "torch", "op": "layer_norm"},
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
