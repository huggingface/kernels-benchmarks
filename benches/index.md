# KERNELS COMMUNITY BENCHMARKS

This report aggregates latency and performance benchmarks across core model components.  
Each section includes:  
- A latency visualization  
- Links to detailed implementation benchmarks  

## TABLE OF CONTENTS
- [METHODOLOGY](#methodology)
- [LAYER NORMALIZATION](#layer-normalization)
- [ROTARY POSITION EMBEDDINGS](#rotary-position-embeddings)
- [FLASH ATTENTION](#flash-attention)
- [CAUSAL CONV1D](#causal-conv1d)
- [ACTIVATION FUNCTIONS](#activation-functions)
- [NOTES](#notes)


## METHODOLOGY

Each benchmark is run with the [Kernels Benchmarking Framework](https://github.com/huggingface/kernels-benchmarks) and follows these principles:  
- a reference implementation (usually PyTorch native) is included for baseline comparison  
- multiple input sizes and batch sizes are tested to reflect real-world usage  
- runs are repeatable via python virtual environments and documented dependencies  
- results are collected and visualized using standardized scripts  

---

<div class="alert">
  <strong>Note:</strong> Latency values are measured in milliseconds (ms). Lower values indicate better performance.
</div>


## LAYER NORMALIZATION

<div class="artifact-preview">
  <img src="layer_norm/results/artifacts/combine/latency.svg" alt="Layer Norm Latency" width="800">
</div>

| Implementation        | Description                        |
| --------------------- | ---------------------------------- |
| HF Kernels Layer Norm | HuggingFace kernels implementation |
| PyTorch Layer Norm    | PyTorch native implementation      |

<p align="center">
  <!-- <button onclick="window.location.href='layer_norm/'" style="margin-left: 20px; padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer;"> -->
  <button 
    onclick="window.location.href='layer_norm/'"
    class="btn">
    Explore Full Bench
  </button>
</p>

---


## ROTARY POSITION EMBEDDINGS

<div class="artifact-preview">
  <img src="rotary/results/artifacts/combine/latency.svg" alt="Rotary Position Embeddings Latency" width="800">
</div>

| Implementation    | Description                        |
| ----------------- | ---------------------------------- |
| HF Kernels Rotary | HuggingFace kernels implementation |
| PyTorch Rotary    | PyTorch native implementation      |

<p align="center">
  <button
    onclick="window.location.href='rotary/'"
    class="btn">
    Explore Full Bench
  </button>
</p>

---


## FLASH ATTENTION

<div class="artifact-preview">
  <img src="flash_attn/results/artifacts/combine/latency.svg" alt="Flash Attention Latency" width="800">
</div>

| Implementation               | Description                               |
| ---------------------------- | ----------------------------------------- |
| Flash Attention              | Flash Attention implementation            |
| HF Kernels Flash Attention   | HuggingFace kernels Flash Attention       |
| HF Kernels Flash Attention 3 | HuggingFace kernels Flash Attention 3     |
| Memory Efficient Attention   | Memory efficient attention implementation |
| Sage Attention               | Sage attention implementation             |
| xFormers                     | xFormers attention implementation         |

<p align="center">
  <button
    onclick="window.location.href='flash_attn/'"
    class="btn">
    Explore Full Bench
  </button>
</p>

---


## CAUSAL CONV1D

<div class="artifact-preview">
  <img src="causal_conv1d/results/artifacts/combine/latency.svg" alt="Causal Conv1D Latency" width="800">
</div>

| Implementation           | Description                        |
| ------------------------ | ---------------------------------- |
| HF Kernels Causal Conv1D | HuggingFace kernels implementation |
| PyTorch Causal Conv1D    | PyTorch native implementation      |

<p align="center">
  <button
    onclick="window.location.href='causal_conv1d/'"
    class="btn">
    Explore Full Bench
  </button>
</p>

---


## ACTIVATION FUNCTIONS

<div class="artifact-preview">
  <img src="activation/results/artifacts/combine/latency.svg" alt="Activation Latency" width="800">
</div>

| Implementation    | Description                               |
| ----------------- | ----------------------------------------- |
| HF Kernels SwiGLU | HuggingFace kernels SwiGLU implementation |
| PyTorch SwiGLU    | PyTorch native SwiGLU implementation      |


<p align="center">
  <button
    onclick="window.location.href='activation/'"
    class="btn">
    Explore Full Bench
  </button>
</p>

---

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
        --border-alert: #001628ff;
    }
    .alert {
        padding: 5px;
        background-color: var(--bg-alert);
        border-left: 6px solid var(--border-alert);
        margin-bottom: 10px;
        border-radius: 6px;
    }
</style>