from __future__ import annotations
from typing import Sequence, Iterable
import torch


def gen_inputs(wl: dict) -> Sequence[torch.Tensor]:
    torch.manual_seed(int(wl.get("seed", 0)))
    batch_size = wl["batch_size"]
    num_queries = wl["num_queries"]
    num_heads = wl["num_heads"]
    embed_dim = wl["embed_dim"]
    num_levels = wl["num_levels"]
    num_points = wl["num_points"]
    spatial_shapes = wl["spatial_shapes"]

    dt = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(wl.get("dtype", "float32"))
    dev = wl.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Convert spatial shapes to tensor
    spatial_shapes_tensor = torch.tensor(spatial_shapes, device=dev, dtype=torch.long)
    num_values = sum(h * w for h, w in spatial_shapes)
    channels = embed_dim // num_heads

    # Create level start indices
    level_start_index = torch.cat(
        [
            torch.tensor([0], device=dev),
            torch.tensor([h * w for h, w in spatial_shapes[:-1]], device=dev).cumsum(0),
        ]
    ).long()

    # Input value tensor: flattened multi-scale features
    value = torch.randn(
        batch_size, num_values, num_heads, channels, device=dev, dtype=dt
    )

    # Sampling locations: normalized coordinates (x, y) in [0, 1]
    sampling_locations = torch.rand(
        batch_size,
        num_queries,
        num_heads,
        num_levels,
        num_points,
        2,
        device=dev,
        dtype=dt,
    )

    # Attention weights: normalized across all sampling points
    attention_weights = torch.rand(
        batch_size, num_queries, num_heads, num_levels, num_points, device=dev, dtype=dt
    )
    attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

    return (
        value,
        spatial_shapes_tensor,
        level_start_index,
        sampling_locations,
        attention_weights,
    )


def ref_impl(inputs: Sequence[torch.Tensor], im2col_step: int = 64) -> torch.Tensor:
    """
    Reference implementation of multi-scale deformable attention.
    Uses vectorized PyTorch operations for reasonable performance.
    """
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights = (
        inputs
    )

    bs, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    _, _, _, channels = value.shape

    output = torch.zeros(
        bs, num_queries, num_heads, channels, device=value.device, dtype=value.dtype
    )

    # Split value tensor by levels
    value_list = value.split([int(h * w) for h, w in spatial_shapes.tolist()], dim=1)

    # Iterate through each level (can't avoid this loop easily)
    for level_idx in range(num_levels):
        h, w = spatial_shapes[level_idx].tolist()
        value_level = value_list[level_idx]  # (bs, h*w, num_heads, channels)

        # Reshape to spatial grid: (bs, num_heads, channels, h, w)
        value_spatial = value_level.reshape(bs, h, w, num_heads, channels).permute(
            0, 3, 4, 1, 2
        )

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
        w_tl = (hh * hw).unsqueeze(
            -1
        )  # top-left: (bs, num_queries, num_heads, num_points, 1)
        w_tr = (hh * lw).unsqueeze(-1)  # top-right
        w_bl = (lh * hw).unsqueeze(-1)  # bottom-left
        w_br = (lh * lw).unsqueeze(-1)  # bottom-right

        # Gather values from the 4 corners using advanced indexing
        batch_idx = (
            torch.arange(bs, device=value.device)
            .view(bs, 1, 1, 1)
            .expand(bs, num_queries, num_heads, num_points)
        )
        head_idx = (
            torch.arange(num_heads, device=value.device)
            .view(1, 1, num_heads, 1)
            .expand(bs, num_queries, num_heads, num_points)
        )

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


def cmp_allclose(
    out: torch.Tensor,
    ref: torch.Tensor,
    rtol=None,
    atol=None,
) -> dict:
    if rtol is None:
        rtol = 3e-3 if out.dtype in (torch.bfloat16, torch.float16) else 1e-5
    if atol is None:
        atol = 5e-3 if out.dtype in (torch.bfloat16, torch.float16) else 1e-5

    diff = (out - ref).abs()

    ok = torch.allclose(out, ref, rtol=rtol, atol=atol) and not (
        torch.isnan(out).any() or torch.isinf(out).any()
    )

    return {
        "ok": bool(ok),
        "rtol": rtol,
        "atol": atol,
        "absmax": float(diff.max().item()),
        "mae": float(diff.mean().item()),
        "mse": float(((out - ref) ** 2).mean().item()),
        "ref": "deformable_detr_torch",
    }


def workloads(dtype="float32", device="cuda") -> Iterable[dict]:
    """Generate workloads for deformable DETR benchmark."""
    # Based on typical Deformable DETR configurations
    for batch_size in [1, 2]:
        for num_queries in [100, 300]:
            for num_heads in [8]:
                for embed_dim in [256]:
                    for num_levels in [4]:
                        for num_points in [4]:
                            # Multi-scale feature map configurations
                            # Common configurations: 32x32, 16x16, 8x8, 4x4
                            spatial_shapes = [[32, 32], [16, 16], [8, 8], [4, 4]]
                            num_values = sum(h * w for h, w in spatial_shapes)

                            yield {
                                "name": f"{device}_B{batch_size}_Q{num_queries}_H{num_heads}_E{embed_dim}_L{num_levels}_P{num_points}",
                                "batch_size": batch_size,
                                "num_queries": num_queries,
                                "num_heads": num_heads,
                                "embed_dim": embed_dim,
                                "num_levels": num_levels,
                                "num_points": num_points,
                                "spatial_shapes": spatial_shapes,
                                "dtype": dtype,
                                "device": device,
                                "seed": 0,
                            }


# single workload for quick testing
def _workloads(dtype="float32", device="cuda") -> Iterable[dict]:
    print("✅ Using single workload for quick testing.")
    spatial_shapes = [[32, 32], [16, 16], [8, 8], [4, 4]]
    yield {
        "name": f"{device}_B1_Q100_H8_E256_L4_P4",
        "batch_size": 1,
        "num_queries": 100,
        "num_heads": 8,
        "embed_dim": 256,
        "num_levels": 4,
        "num_points": 4,
        "spatial_shapes": spatial_shapes,
        "dtype": dtype,
        "device": device,
        "seed": 0,
    }
