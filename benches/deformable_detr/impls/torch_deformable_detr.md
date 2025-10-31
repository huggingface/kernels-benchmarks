---
on_github: huggingface/kernels-benchmarks
---

# PyTorch Native - Deformable DETR

## GPU Info

```python id=nv
import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
```

## Deformable DETR Multi-Scale Deformable Attention Benchmark (PyTorch Native)

```python id=benchmark outputs=deformable_detr.jsonl
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
import sys
from kernels_benchmark_tools import KernelTypeEnum, run_benchmark


def torch_deformable_detr(
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step=64
):
    """
    PyTorch native reference implementation of multi-scale deformable attention.
    Uses vectorized bilinear interpolation for reasonable performance.
    """
    bs, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    _, _, _, channels = value.shape

    output = torch.zeros(bs, num_queries, num_heads, channels, device=value.device, dtype=value.dtype)

    # Split value tensor by levels
    value_list = value.split([int(h * w) for h, w in spatial_shapes.tolist()], dim=1)

    # Iterate through each level (can't avoid this loop easily)
    for level_idx in range(num_levels):
        h, w = spatial_shapes[level_idx].tolist()
        value_level = value_list[level_idx]  # (bs, h*w, num_heads, channels)

        # Reshape to spatial grid: (bs, num_heads, channels, h, w)
        value_spatial = value_level.reshape(bs, h, w, num_heads, channels).permute(0, 3, 4, 1, 2)

        # Get sampling locations and weights for this level
        # loc: (bs, num_queries, num_heads, num_points, 2)
        loc = sampling_locations[:, :, :, level_idx, :, :]
        # weight: (bs, num_queries, num_heads, num_points)
        weight = attention_weights[:, :, :, level_idx, :]

        # Convert normalized coordinates to pixel coordinates
        # loc[..., 0] is x (width), loc[..., 1] is y (height)
        x = loc[..., 0] * w - 0.5  # (bs, num_queries, num_heads, num_points)
        y = loc[..., 1] * h - 0.5

        # Get integer coordinates for bilinear interpolation
        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()
        x1 = x0 + 1
        y1 = y0 + 1

        # Compute interpolation weights BEFORE clamping (important!)
        lw = x - x0.float()  # weight for x direction
        lh = y - y0.float()  # weight for y direction
        hw = 1 - lw
        hh = 1 - lh

        # Create mask for valid sample locations
        valid = (y > -1) & (x > -1) & (y < h) & (x < w)

        # Create masks for each corner being in bounds
        mask_tl = ((y0 >= 0) & (x0 >= 0)).unsqueeze(-1).float()
        mask_tr = ((y0 >= 0) & (x1 <= w - 1)).unsqueeze(-1).float()
        mask_bl = ((y1 <= h - 1) & (x0 >= 0)).unsqueeze(-1).float()
        mask_br = ((y1 <= h - 1) & (x1 <= w - 1)).unsqueeze(-1).float()

        # Clamp coordinates for safe indexing
        x0_clamped = torch.clamp(x0, 0, w - 1)
        x1_clamped = torch.clamp(x1, 0, w - 1)
        y0_clamped = torch.clamp(y0, 0, h - 1)
        y1_clamped = torch.clamp(y1, 0, h - 1)

        # Bilinear interpolation weights for all 4 corners
        w_tl = (hh * hw).unsqueeze(-1)  # top-left: (bs, num_queries, num_heads, num_points, 1)
        w_tr = (hh * lw).unsqueeze(-1)  # top-right
        w_bl = (lh * hw).unsqueeze(-1)  # bottom-left
        w_br = (lh * lw).unsqueeze(-1)  # bottom-right

        # Gather values from the 4 corners using advanced indexing
        batch_idx = torch.arange(bs, device=value.device).view(bs, 1, 1, 1).expand(bs, num_queries, num_heads, num_points)
        head_idx = torch.arange(num_heads, device=value.device).view(1, 1, num_heads, 1).expand(bs, num_queries, num_heads, num_points)

        # Gather corner values with clamped indices, then apply corner masks
        v_tl = value_spatial[batch_idx, head_idx, :, y0_clamped, x0_clamped] * mask_tl
        v_tr = value_spatial[batch_idx, head_idx, :, y0_clamped, x1_clamped] * mask_tr
        v_bl = value_spatial[batch_idx, head_idx, :, y1_clamped, x0_clamped] * mask_bl
        v_br = value_spatial[batch_idx, head_idx, :, y1_clamped, x1_clamped] * mask_br

        # Bilinear interpolation
        sampled = w_tl * v_tl + w_tr * v_tr + w_bl * v_bl + w_br * v_br

        # Apply valid mask (only accumulate if entire sample location is valid)
        sampled = sampled * valid.unsqueeze(-1).float()

        # Apply attention weights and sum over points
        # weight: (bs, num_queries, num_heads, num_points)
        # Expand weight: (bs, num_queries, num_heads, num_points, 1)
        weighted_sampled = sampled * weight.unsqueeze(-1)

        # Sum over points: (bs, num_queries, num_heads, channels)
        output += weighted_sampled.sum(dim=3)

    # Flatten last two dimensions to match kernel output
    return output.reshape(bs, num_queries, num_heads * channels)


run_benchmark(
    kernel_type=KernelTypeEnum.DEFORMABLE_DETR,
    impl_name="torch_eager",
    impl_tags={"family": "pytorch", "backend": "eager"},
    impl_func=torch_deformable_detr,
    dtype="float32",
)
```
