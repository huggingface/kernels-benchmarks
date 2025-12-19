---
on_github: huggingface/kernels-benchmarks
on_huggingface: drbh/yamoe
platforms:
  - linux
---

# GptOssExperts - OpenAI-style MoE

## GPU Info

```python id=nv
import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
```

## OpenAI-style MoE Benchmark (GptOssExperts Reference)

```python id=benchmark outputs=openai_moe.jsonl timeout=600
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
# kernels-benchmark-tools = { path = "../../../../../tools", editable = true }
# kernels = { git = "https://github.com/huggingface/kernels.git" }
# ///
import torch
import sys
from kernels_benchmark_tools import KernelTypeEnum, run_benchmark
from kernels import get_kernel

# Load yamoe to get GptOssExperts reference
yamoe = get_kernel("drbh/yamoe", revision="v0.2.0")
GptOssExperts = yamoe.vendored.gpt_oss_mlp.GptOssExperts


def gpt_oss_openai_moe(
    hidden_states,
    router_indices,
    routing_weights,
    gate_up_proj,
    gate_up_proj_bias,
    down_proj,
    down_proj_bias,
):
    """
    GptOssExperts reference implementation of OpenAI-style MoE.
    This is the reference model implementation from the original GPT OSS codebase.
    """
    B, S, H = hidden_states.shape
    E = routing_weights.shape[2]

    # Create a config object for GptOssExperts
    config = type("Config", (), {})()
    config.hidden_size = H
    config.intermediate_size = gate_up_proj.shape[2] // 2  # expert_dim / 2 = H
    config.num_local_experts = E

    # Initialize model
    model = GptOssExperts(config)

    # Set weights from benchmark inputs
    model.gate_up_proj.data = gate_up_proj
    model.gate_up_proj_bias.data = gate_up_proj_bias
    model.down_proj.data = down_proj
    model.down_proj_bias.data = down_proj_bias

    model = model.to(hidden_states.device)
    model.eval()

    # Force GptOssExperts to use CPU path for correctness (matches naive_moe_ref behavior)
    # The GPU path processes all experts which can lead to numerical differences
    # CPU path explicitly uses router_indices like the reference implementation
    model.train()  # Force CPU path

    # Flatten routing_weights to [batch_seq, num_experts]
    routing_weights_flat = routing_weights.view(-1, E)

    # Run forward pass
    with torch.no_grad():
        output = model(hidden_states, router_indices, routing_weights_flat)

    model.eval()  # Reset to eval mode

    return output


run_benchmark(
    kernel_type=KernelTypeEnum.OPENAI_MOE,
    impl_name="gpt_oss_experts",
    impl_tags={"family": "reference", "backend": "pytorch"},
    impl_func=gpt_oss_openai_moe,
    dtype="float32",
)
```
