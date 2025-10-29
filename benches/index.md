
# All Benchmarks Aggregated Report

## [Layer Norm](layer_norm/)

<div class="artifact-preview">
<img src="layer_norm/results/artifacts/combine/latency.svg" alt="Layer Norm Latency" width="800">
</div>

| Implementation | Description |
|----------------|-------------|
| [HF Kernels Layer Norm](layer_norm/impls/hf_kernels_layer_norm.html) | HuggingFace kernels implementation |
| [PyTorch Layer Norm](layer_norm/impls/torch_layer_norm.html) | PyTorch native implementation |

## [Rotary Position Embeddings](rotary/)

<div class="artifact-preview">
<img src="rotary/results/artifacts/combine/latency.svg" alt="Rotary Position Embeddings Latency" width="800">
</div>

| Implementation | Description |
|----------------|-------------|
| [HF Kernels Rotary](rotary/impls/hf_kernels_rotary.html) | HuggingFace kernels implementation |
| [PyTorch Rotary](rotary/impls/torch_rotary.html) | PyTorch native implementation |

## [Flash Attention](flash_attn/)

<div class="artifact-preview">
<img src="flash_attn/results/artifacts/combine/latency.svg" alt="Flash Attention Latency" width="800">
</div>

| Implementation | Description |
|----------------|-------------|
| [Flash Attention](flash_attn/impls/flash_attention.html) | Flash Attention implementation |
| [HF Kernels Flash Attention](flash_attn/impls/hf_kernels_flash_attn.html) | HuggingFace kernels Flash Attention |
| [HF Kernels Flash Attention 3](flash_attn/impls/hf_kernels_flash_attn3.html) | HuggingFace kernels Flash Attention 3 |
| [Memory Efficient Attention](flash_attn/impls/mem_efficient_attention.html) | Memory efficient attention implementation |
| [Sage Attention](flash_attn/impls/sage_attention.html) | Sage attention implementation |
| [xFormers](flash_attn/impls/xformers.html) | xFormers attention implementation |

## [Causal Conv1D](causal_conv1d/)

<div class="artifact-preview">
<img src="causal_conv1d/results/artifacts/combine/latency.svg" alt="Causal Conv1D Latency" width="800">
</div>

| Implementation | Description |
|----------------|-------------|
| [HF Kernels Causal Conv1D](causal_conv1d/impls/hf_kernels_causal_conv1d.html) | HuggingFace kernels implementation |
| [PyTorch Causal Conv1D](causal_conv1d/impls/torch_causal_conv1d.html) | PyTorch native implementation |

## [Activation](activation/)

<div class="artifact-preview">
<img src="activation/results/artifacts/combine/latency.svg" alt="Activation Latency" width="800">
</div>

| Implementation | Description |
|----------------|-------------|
| [HF Kernels SwiGLU](activation/impls/hf_kernels_swiglu.html) | HuggingFace kernels SwiGLU implementation |
| [PyTorch SwiGLU](activation/impls/torch_swiglu.html) | PyTorch native SwiGLU implementation |

## [ReLU](relu/)

<div class="artifact-preview">
<img src="relu/results/artifacts/combine/latency.svg" alt="ReLU Latency" width="800">
</div>

| Implementation | Description |
|----------------|-------------|
| [HF Kernels ReLU](relu/impls/hf_kernels_relu.html) | HuggingFace kernels ReLU implementation |
| [PyTorch ReLU](relu/impls/torch_relu.html) | PyTorch native ReLU implementation |

