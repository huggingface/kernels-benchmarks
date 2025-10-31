<div class="linkbar">
<a target="_blank" href="https://github.com/huggingface/kernels">Python Library</a> |
<a target="_blank" href="https://github.com/huggingface/kernel-builder">Builder</a> |
<a target="_blank" href="https://github.com/huggingface/kernels-community">Community</a> |
<a target="_blank" href="https://huggingface.co/kernels-community">Community Hub</a> |
<a target="_blank" href="https://github.com/huggingface/kernels-benchmarks">Benchmarks</a>
</div>

<br/>

# KERNELS COMMUNITY BENCHMARKS

This report aggregates latency and performance benchmarks across core model components.  
Each section includes:  
- A latency visualization  
- Links to detailed implementation benchmarks  

## TABLE OF CONTENTS
- [ACTIVATION FUNCTIONS](#activation-functions)
- [FLASH ATTENTION](#flash-attention)
- [DEFORMABLE DETR](#deformable-detr)
- [OPENAI-STYLE MOE](#openai-style-moe)
- [ROTARY POSITION EMBEDDINGS](#rotary-position-embeddings)
- [CAUSAL CONV1D](#causal-conv1d)
- [LAYER NORMALIZATION](#layer-normaliz=ation)


## RUN YOURSELF

To run the benchmarks locally, clone the repository and use `uvx` to build and run the benchmarks:

Note benches are made to run on a machine with a compatible NVIDIA GPU and CUDA installed, other hardware may not not work as expected.

```bash
git clone https://github.com/huggingface/kernels-benchmarks.git
cd kernels-benchmarks
uvx https://github.com/drbh/uvnote.git build benches
```



## METHODOLOGY

Each benchmark is run with the
<a target="_blank" href="https://github.com/huggingface/kernels-benchmarks">Kernels Benchmarking Framework</a> and follows these principles:  
- a reference implementation (usually PyTorch native) is included for baseline comparison  
- multiple input sizes and batch sizes are tested to reflect real-world usage  
- runs are repeatable via python virtual environments and documented dependencies  
- results are collected and visualized using standardized scripts  


<br/>

## BENCHMARKS

<div class="alert">
  <strong>Note:</strong> Latency values are measured in milliseconds (ms). Lower values indicate better performance.
</div>



## ACTIVATION FUNCTIONS

<div class="artifact-preview">
  <img src="activation/results/artifacts/combine/latency.svg" alt="Activation Latency" width="800">
</div>

| Implementation    | Description                               | Source                                                                          | HF                                                        | Bench                                            |
| ----------------- | ----------------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------ |
| HF Kernels SwiGLU | HuggingFace kernels SwiGLU implementation | [GitHub](https://github.com/huggingface/kernels-community/tree/main/activation) | [HF](https://huggingface.co/kernels-community/activation) | [Bench](activation/impls/hf_kernels_swiglu.html) |
| PyTorch SwiGLU    | PyTorch native SwiGLU implementation      | -                                                                               | -                                                         | [Bench](activation/impls/torch_swiglu.html)      |


<p align="center">
  <button
    onclick="window.location.href='/#/activation/'"
    class="btn">
    Explore Full Bench
  </button>
</p>


---


## FLASH ATTENTION

<div class="artifact-preview">
  <img src="flash_attn/results/artifacts/combine/latency.svg" alt="Flash Attention Latency" width="800">
</div>

| Implementation               | Description                               | Source                                                                           | HF                                                            | Bench                                                  |
| ---------------------------- | ----------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------ |
| Flash Attention              | Torch SDPA Flash Attention implementation | -                                                                                | -                                                             | [Bench](flash_attn/impls/flash_attention.html)         |
| HF Kernels Flash Attention 2 | HuggingFace kernels Flash Attention       | [GitHub](https://github.com/huggingface/kernels-community/tree/main/flash-attn2) | [HF](https://huggingface.co/kernels-community/flash-attn2)    | [Bench](flash_attn/impls/hf_kernels_flash_attn.html)   |
| HF Kernels Flash Attention 3 | HuggingFace kernels Flash Attention 3     | [GitHub](https://github.com/huggingface/kernels-community/tree/main/flash-attn3) | [HF](https://huggingface.co/kernels-community/flash-attn3)    | [Bench](flash_attn/impls/hf_kernels_flash_attn3.html)  |
| Memory Efficient Attention   | Memory efficient attention implementation |                                                                                  | -                                                             | [Bench](flash_attn/impls/mem_efficient_attention.html) |
| Sage Attention               | Sage attention implementation             |                                                                                  | [HF](https://huggingface.co/kernels-community/sage_attention) | [Bench](flash_attn/impls/sage_attention.html)          |
| xFormers                     | xFormers attention implementation         | [GitHub](https://github.com/facebookresearch/xformers)                           | -                                                             | [Bench](flash_attn/impls/xformers.html)                |

<p align="center">
  <button
    onclick="window.location.href='flash_attn/'"
    class="btn">
    Explore Full Bench
  </button>
</p>

---


## DEFORMABLE DETR

<div class="artifact-preview">
  <img src="deformable_detr/results/artifacts/combine/latency.svg" alt="Deformable DETR Latency" width="800">
</div>

| Implementation             | Description                                        | Source                                                                               | HF                                                             | Bench                                                          |
| -------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------- | -------------------------------------------------------------- |
| HF Kernels Deformable DETR | HuggingFace kernels Deformable DETR implementation | [GitHub](https://github.com/huggingface/kernels-community/tree/main/deformable-detr) | [HF](https://huggingface.co/kernels-community/deformable-detr) | [Bench](deformable_detr/impls/hf_kernels_deformable_detr.html) |
| PyTorch Deformable DETR    | PyTorch native Deformable DETR implementation      | -                                                                                    | -                                                              | [Bench](deformable_detr/impls/torch_deformable_detr.html)      |

<p align="center">
  <button
    onclick="window.location.href='deformable_detr/'"
    class="btn">
    Explore Full Bench
  </button>
</p>


---


## OPENAI-STYLE MOE

<div class="artifact-preview">
  <img src="openai_moe/results/artifacts/combine/latency.svg" alt="OpenAI MoE Latency" width="800">
</div>

| Implementation | Description                                    | Source | HF  | Bench                                       |
| -------------- | ---------------------------------------------- | ------ | --- | ------------------------------------------- |
| GptOssExperts  | GPT OSS reference OpenAI-style MoE             |        |     | [Bench](openai_moe/impls/gpt_oss_moe.html)  |
| Binned PyTorch | Binned PyTorch OpenAI-style MoE implementation | -      | -   | [Bench](openai_moe/impls/binned_torch.html) |

<p align="center">
  <button
    onclick="window.location.href='openai_moe/'"
    class="btn">
    Explore Full Bench
  </button>
</p>


---


## CAUSAL CONV1D

<div class="artifact-preview">
  <img src="causal_conv1d/results/artifacts/combine/latency.svg" alt="Causal Conv1D Latency" width="800">
</div>

| Implementation           | Description                        | Source                                                                             | HF                                                           | Bench                                                      |
| ------------------------ | ---------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| HF Kernels Causal Conv1D | HuggingFace kernels implementation | [GitHub](https://github.com/huggingface/kernels-community/tree/main/causal-conv1d) | [HF](https://huggingface.co/kernels-community/causal-conv1d) | [Bench](causal_conv1d/impls/hf_kernels_causal_conv1d.html) |
| PyTorch Causal Conv1D    | PyTorch native implementation      | -                                                                                  | -                                                            | [Bench](causal_conv1d/impls/torch_causal_conv1d.html)      |

<p align="center">
  <button
    onclick="window.location.href='causal_conv1d/'"
    class="btn">
    Explore Full Bench
  </button>
</p>


---


## ROTARY POSITION EMBEDDINGS

<div class="artifact-preview">
  <img src="rotary/results/artifacts/combine/latency.svg" alt="Rotary Position Embeddings Latency" width="800">
</div>

| Implementation    | Description                        | Source                                                                      | HF                                                    | Bench                                        |
| ----------------- | ---------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------- | -------------------------------------------- |
| HF Kernels Rotary | HuggingFace kernels implementation | [GitHub](https://github.com/huggingface/kernels-community/tree/main/rotary) | [HF](https://huggingface.co/kernels-community/rotary) | [Bench](rotary/impls/hf_kernels_rotary.html) |
| PyTorch Rotary    | PyTorch native implementation      | -                                                                           | -                                                     | [Bench](rotary/impls/torch_rotary.html)      |

<p align="center">
  <button
    onclick="window.location.href='rotary/'"
    class="btn">
    Explore Full Bench
  </button>
</p>


---


## LAYER NORMALIZATION

<div class="artifact-preview">
  <img src="layer_norm/results/artifacts/combine/latency.svg" alt="Layer Norm Latency" width="800">
</div>

| Implementation        | Description                        | Source                                                                          | HF                                                        | Bench                                                |
| --------------------- | ---------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------- |
| HF Kernels Layer Norm | HuggingFace kernels implementation | [GitHub](https://github.com/huggingface/kernels-community/tree/main/layer-norm) | [HF](https://huggingface.co/kernels-community/layer-norm) | [Bench](layer_norm/impls/hf_kernels_layer_norm.html) |
| PyTorch Layer Norm    | PyTorch native implementation      | -                                                                               | -                                                         | [Bench](layer_norm/impls/torch_layer_norm.html)      |

<p align="center">
  <button 
    onclick="window.location.href='layer_norm/'"
    class="btn">
    Explore Full Bench
  </button>
</p>

<style>
    .controls {
        display: none !important;
    }
    .status-widget {
        display: none !important;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
        background: var(--bg-secondary);
        border: 1px solid var(--border-primary);
        border-radius: 1px;
    }
    table th,
    table td {
        padding: 0.5rem 0.75rem;
        text-align: left;
        border: 1px solid var(--border-primary);
    }
    table th {
        background: var(--bg-tertiary);
        font-weight: 600;
        color: var(--text-primary);
    }
    table tbody tr:hover {
        background: var(--bg-artifact-hover);
    }
    .btn {
        margin: 10px 0;
        padding: 10px 20px;
        background-color: transparent;
        color: inherit;
        border: 1px solid var(--text-primary);
        border-radius: 5px;
        cursor: pointer;
    }
    .btn:hover {
        background-color: var(--bg-artifact-hover);
    }
    :root {
        --bg-alert: #0069cbff;
    }
    .alert {
        padding: 5px 10px;
        background-color: var(--bg-alert);
        margin-bottom: 10px;
        border-radius: 6px;
    }
</style>