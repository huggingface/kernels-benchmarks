import json, sys, time, collections as C
from typing import List, Dict, Any, Optional


def detailed_tensor_comparison(out, ref, rtol=1e-3, atol=1e-3, name="Tensor"):
    """
    Compare two tensors and print detailed diagnostics on failure.

    Args:
        out: Output tensor from implementation
        ref: Reference tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        name: Name for the comparison (for logging)

    Returns:
        Dictionary with comparison metrics
    """
    try:
        import torch
    except ImportError:
        # Fallback if torch not available (shouldn't happen in practice)
        return {"ok": False, "error": "torch not available for comparison"}

    # Compute differences
    diff = (out - ref).abs()
    rel_diff = diff / (ref.abs() + 1e-8)  # Add epsilon to avoid division by zero

    # Check for NaN/Inf
    has_nan_out = torch.isnan(out).any()
    has_inf_out = torch.isinf(out).any()
    has_nan_ref = torch.isnan(ref).any()
    has_inf_ref = torch.isinf(ref).any()

    # Perform comparison
    ok = torch.allclose(out, ref, rtol=rtol, atol=atol) and not (
        has_nan_out or has_inf_out
    )

    # Compute metrics
    metrics = {
        "ok": bool(ok),
        "rtol": rtol,
        "atol": atol,
        "absmax": float(diff.max().item()),
        "mae": float(diff.mean().item()),
        "mse": float(((out - ref) ** 2).mean().item()),
        "relmax": float(rel_diff.max().item()),
    }

    # Print detailed diagnostics if comparison failed
    if not ok:
        print("\n" + "=" * 70)
        print(f"❌ COMPARISON FAILED: {name}")
        print("=" * 70)

        # Shape information
        print(f"\nShape Information:")
        print(f"  Output shape: {tuple(out.shape)}")
        print(f"  Reference shape: {tuple(ref.shape)}")
        if out.shape != ref.shape:
            print(f"  ⚠️  SHAPE MISMATCH!")

        # Dtype information
        print(f"\nData Type Information:")
        print(f"  Output dtype: {out.dtype}")
        print(f"  Reference dtype: {ref.dtype}")
        if out.dtype != ref.dtype:
            print(f"  ⚠️  DTYPE MISMATCH!")

        # NaN/Inf checks
        print(f"\nNaN/Inf Checks:")
        print(f"  Output has NaN: {has_nan_out}")
        print(f"  Output has Inf: {has_inf_out}")
        print(f"  Reference has NaN: {has_nan_ref}")
        print(f"  Reference has Inf: {has_inf_ref}")

        if has_nan_out or has_inf_out:
            print(f"  ⚠️  Output contains invalid values!")
            if has_nan_out:
                nan_count = torch.isnan(out).sum().item()
                print(
                    f"     NaN count: {nan_count} / {out.numel()} ({100 * nan_count / out.numel():.2f}%)"
                )
            if has_inf_out:
                inf_count = torch.isinf(out).sum().item()
                print(
                    f"     Inf count: {inf_count} / {out.numel()} ({100 * inf_count / out.numel():.2f}%)"
                )

        # Tolerance information
        print(f"\nTolerance Thresholds:")
        print(f"  Relative tolerance: {rtol}")
        print(f"  Absolute tolerance: {atol}")

        # Difference statistics
        print(f"\nDifference Statistics:")
        print(f"  Max absolute diff: {metrics['absmax']:.6e}")
        print(f"  Mean absolute diff: {metrics['mae']:.6e}")
        print(f"  RMS error: {metrics['mse'] ** 0.5:.6e}")
        print(f"  Max relative diff: {metrics['relmax']:.6e}")

        # Percentile analysis (convert to float32 for quantile computation)
        diff_flat = diff.flatten()
        if len(diff_flat) > 0:
            # quantile requires float32/float64, so convert if needed
            diff_for_quantile = (
                diff_flat.to(torch.float32)
                if diff_flat.dtype in [torch.bfloat16, torch.float16]
                else diff_flat
            )
            print(f"\nDifference Percentiles:")
            print(f"  p50: {torch.median(diff_for_quantile).item():.6e}")
            print(f"  p90: {torch.quantile(diff_for_quantile, 0.9).item():.6e}")
            print(f"  p99: {torch.quantile(diff_for_quantile, 0.99).item():.6e}")
            print(f"  p99.9: {torch.quantile(diff_for_quantile, 0.999).item():.6e}")

        # Find worst mismatches
        print(f"\nWorst Mismatches (top 5 by absolute difference):")
        flat_diff = diff.flatten()
        flat_out = out.flatten()
        flat_ref = ref.flatten()

        topk = min(5, len(flat_diff))
        worst_indices = torch.topk(flat_diff, topk).indices

        for i, idx in enumerate(worst_indices, 1):
            idx_val = idx.item()
            out_val = flat_out[idx_val].item()
            ref_val = flat_ref[idx_val].item()
            diff_val = flat_diff[idx_val].item()
            rel_val = diff_val / (abs(ref_val) + 1e-8)
            print(
                f"  #{i}: output={out_val:.6e}, ref={ref_val:.6e}, diff={diff_val:.6e}, rel={rel_val:.6e}"
            )

        # Value range comparison
        print(f"\nValue Ranges:")
        print(f"  Output: [{out.min().item():.6e}, {out.max().item():.6e}]")
        print(f"  Reference: [{ref.min().item():.6e}, {ref.max().item():.6e}]")

        # Suggestions
        print(f"\n💡 Debugging Suggestions:")

        # Dtype-specific precision limits
        if out.dtype == torch.bfloat16:
            print(f"  • Using bfloat16 (limited precision: ~7-8 bits mantissa)")
            print(f"    → Expected precision: ~0.78% (rtol ~1e-2)")
            print(
                f"    → Max relative diff {metrics['relmax']:.2e} vs precision limit ~1e-2"
            )
            if metrics["relmax"] < 2e-2:
                print(f"    → Diff is within expected bfloat16 precision range")
        elif out.dtype == torch.float16:
            print(f"  • Using float16 (limited precision: ~10 bits mantissa)")
            print(f"    → Expected precision: ~0.1% (rtol ~5e-3)")

        if metrics["relmax"] < 0.1 and metrics["absmax"] > atol:
            print(f"  • Large absolute differences but small relative differences")
            print(f"    → Consider increasing absolute tolerance (atol)")
        if metrics["absmax"] < atol * 10 and not ok:
            print(f"  • Close to passing tolerance")
            print(f"    → Check if dtype precision is adequate ({out.dtype})")
        if has_nan_out or has_inf_out:
            print(f"  • Invalid values detected")
            print(f"    → Check for division by zero or overflow in implementation")
        if out.dtype != ref.dtype:
            print(f"  • Dtype mismatch detected")
            print(f"    → Ensure implementation uses same dtype as reference")

        print("=" * 70 + "\n")

    return metrics


def load_jsonl(paths: List[str]) -> List[Dict[str, Any]]:
    return [json.loads(l) for p in paths for l in open(p)]


def summarize(jsonl_files: List[str]):
    rows = load_jsonl(jsonl_files)
    g = C.defaultdict(list)
    failed = C.defaultdict(list)

    for r in rows:
        key = (r["impl"], r["wl"]["name"])
        if r.get("lat_ms"):
            g[key].append(r)
        else:
            failed[key].append(r)

    keys = sorted(set(g.keys()) | set(failed.keys()))
    print(f"{'impl':24} {'wl':18}  p50(ms)  ok")

    for k in keys:
        impl, wl = k
        if k in g:
            rs = g[k]
            p50 = sum(r["lat_ms"]["p50"] for r in rs) / len(rs)
            ok = all(r.get("ok", False) for r in rs)
            print(f"{impl:24} {wl:18}  {p50:7.2f}  {str(ok):>3}")
        elif k in failed:
            rs = failed[k]
            err_msg = rs[0].get("err", {}).get("msg", "Unknown error")
            print(f"{impl:24} {wl:18}    FAIL  False")
            print(f"  Error: {err_msg}")


def outliers(jsonl_files: List[str], top_n: int = 3):
    rows = load_jsonl(jsonl_files)

    outliers = []
    for r in rows:
        if r.get("lat_ms") and r.get("ok"):
            p50 = r["lat_ms"]["p50"]
            p90 = r["lat_ms"]["p90"]
            outlier_score = p90 / p50 if p50 > 0 else 0
            outliers.append((outlier_score, r))

    outliers.sort(key=lambda x: x[0], reverse=True)

    print(f"Top {top_n} outlier runs (p90/p50 ratio):")
    print(f"{'ratio':>6} {'impl':20} {'wl':15} {'p50':>8} {'p90':>8}")

    for score, r in outliers[:top_n]:
        impl = r["impl"]
        wl = r["wl"]["name"]
        p50 = r["lat_ms"]["p50"]
        p90 = r["lat_ms"]["p90"]
        print(f"{score:6.2f} {impl:20} {wl:15} {p50:8.2f} {p90:8.2f}")


def get_gpu_power() -> Optional[float]:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        return power_mw / 1000.0
    except (ImportError, Exception):
        return None


def power():
    power = get_gpu_power()
    if power:
        print(f"Current GPU power: {power:.1f}W")
    else:
        print("GPU power monitoring not available (need nvidia-ml-py + NVIDIA GPU)")


def plot_latency(records: List[Dict[str, Any]], save_path: str = "latency.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required")

    valid_records = [r for r in records if r.get("ok") and r.get("lat_ms")]
    if not valid_records:
        print("No valid records found")
        return

    impls = {}
    for record in valid_records:
        impl = record["impl"]
        wl_name = record["wl"]["name"]
        p50 = record["lat_ms"]["p50"]

        if impl not in impls:
            impls[impl] = {"workloads": [], "p50": []}
        impls[impl]["workloads"].append(wl_name)
        impls[impl]["p50"].append(p50)

    plt.figure(figsize=(12, 8))
    for impl, data in impls.items():
        plt.plot(data["workloads"], data["p50"], marker="o", label=impl)

    plt.xlabel("Workload")
    plt.ylabel("Latency P50 (ms)")
    plt.title("Attention Implementation Latency")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved {save_path}")


def viz(jsonl_files: List[str]):
    records = load_jsonl(jsonl_files)
    print(f"Loaded {len(records)} records")
    plot_latency(records)


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools.py <command> [files...]")
        print("Commands: summarize, outliers, power, viz")
        return

    cmd = sys.argv[1]
    files = sys.argv[2:] if len(sys.argv) > 2 else ["attn.jsonl"]

    if cmd == "summarize":
        summarize(files)
    elif cmd == "outliers":
        outliers(files)
    elif cmd == "power":
        power()
    elif cmd == "viz":
        viz(files)
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
