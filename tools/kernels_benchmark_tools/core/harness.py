from __future__ import annotations
import json, os, time, uuid, statistics as stats, platform
from typing import Any, Callable, Dict, Iterable, List, Sequence
import torch
from torch.utils.benchmark import Timer
from torch.profiler import profile, ProfilerActivity, record_function

IMPLS: Dict[str, Callable] = {}


def add(name: str, fn: Callable, tags: Dict[str, str] | None = None):
    IMPLS[name] = fn
    IMPLS[name]._tags = tags or {}
    return fn


def _env_block() -> dict:
    cuda = torch.version.cuda or ""
    gpu, sm = "", ""
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gpu, sm = p.name, f"{p.major}.{p.minor}"
    return {
        "torch": torch.__version__,
        "cuda": cuda,
        "gpu": gpu,
        "sm": sm,
        "py": platform.python_version(),
        "plat": platform.platform(),
    }


def _reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak():
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return None


def _quantiles(xs: List[float], qs=(0.10, 0.50, 0.90)) -> tuple[float, ...]:
    if not xs:
        return tuple(float("nan") for _ in qs)
    xs = sorted(xs)
    n = len(xs) - 1
    return tuple(float(xs[max(0, min(n, int(q * n)))]) for q in qs)


def run(
    workloads: Iterable[dict],
    *,
    jsonl: str = "bench.jsonl",
    reps: int = 10,
    warmup: int = 3,
    gen: Callable[[dict], Sequence[Any]],
    ref: Callable[[Sequence[Any]], Any],
    cmp: Callable[[Any, Any], dict],
    include: List[str] | None = None,
    exclude: List[str] | None = None,
    profile_trace: bool = False,
):
    """
    Run benchmarks on a set of workloads and implementations.

    Uses torch.utils.benchmark.Timer for timing measurements, which provides:
    - Automatic CUDA synchronization (handles torch.cuda.synchronize() internally)
    - Robust timing across CPU/CUDA devices
    - Proper handling of edge cases (streams, autograd state, etc.)

    Additional metadata captured beyond basic timing:
    - IQR (interquartile range) for variance analysis
    - raw_times: full array of all timing measurements
    - has_warnings: flag for high variance (IQR > 10% of median)
    - compile_ms: separate first-run compilation time
    - peak_bytes: peak CUDA memory usage

    Args:
        profile_trace: If True, run torch.profiler on each implementation and print
                       trace summary to stdout (no files saved). Shows CPU/CUDA time,
                       kernel launches, and memory operations.
    """
    os.makedirs(os.path.dirname(jsonl) or ".", exist_ok=True)
    run_id = uuid.uuid4().hex
    names = list(IMPLS.keys())
    if include:
        names = [n for n in names if n in set(include)]
    if exclude:
        names = [n for n in names if n not in set(exclude)]
    env = _env_block()
    now = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for wl in workloads:
        inputs = gen(wl)
        ref_out = ref(inputs)
        for name in names:
            fn = IMPLS[name]
            try:
                _reset_peak()

                # Compile time measurement (first call, handled by Timer with sync)
                timer = Timer(
                    stmt="fn(*inputs)",
                    globals={"fn": fn, "inputs": inputs},
                    label=name,
                    sub_label=str(wl.get("name", "")),
                    description=f"{wl.get('dtype', '')}",
                )
                t_compile = timer.timeit(1)
                compile_ms = t_compile.mean * 1000

                # Warmup runs
                if warmup > 0:
                    timer.timeit(warmup)

                # Main timing: individual reps (Timer handles CUDA sync automatically)
                times = []
                for _ in range(reps):
                    t = timer.timeit(1)
                    times.append(t.mean * 1000)  # Convert to milliseconds

                # Correctness check
                out = fn(*inputs)
                corr = cmp(out, ref_out)

                # Optional profiling trace (printed to stdout, no files saved)
                if profile_trace:
                    activities = [ProfilerActivity.CPU]
                    if torch.cuda.is_available():
                        activities.append(ProfilerActivity.CUDA)

                    print(f"\n{'=' * 70}")
                    print(f"PROFILE TRACE: {name} | {wl.get('name', 'workload')}")
                    print(f"{'=' * 70}")

                    with profile(
                        activities=activities,
                        record_shapes=True,
                        with_stack=False,
                    ) as prof:
                        with record_function(f"{name}"):
                            for _ in range(3):  # Profile 3 runs for stable stats
                                fn(*inputs)

                    # Print key_averages table to stdout
                    print(
                        prof.key_averages().table(
                            sort_by="cuda_time_total"
                            if torch.cuda.is_available()
                            else "cpu_time_total",
                            row_limit=20,
                        )
                    )
                    print()

                # Compute statistics
                p10, p50, p90 = _quantiles(times, (0.10, 0.50, 0.90))
                q25, q75 = _quantiles(times, (0.25, 0.75))[:2]
                iqr = q75 - q25

                # Check for high variance (similar to Timer's has_warnings)
                has_warnings = False
                if times and p50 > 0:
                    cv = iqr / p50  # Coefficient of variation
                    has_warnings = cv > 0.1  # Warn if IQR > 10% of median

                rec = {
                    "ts": now(),
                    "run": run_id,
                    "impl": name,
                    "tags": getattr(fn, "_tags", {}),
                    "wl": wl,
                    "env": env,
                    "lat_ms": {
                        "p10": p10,
                        "p50": p50,
                        "p90": p90,
                        "mean": float(stats.fmean(times)),
                        "iqr": iqr,  # NEW: Interquartile range
                        "raw_times": times,  # NEW: All raw timing data
                        "has_warnings": has_warnings,  # NEW: High variance warning
                        "reps": reps,
                        "warmup": warmup,
                    },
                    "compile_ms": compile_ms,
                    "peak_bytes": _peak(),
                    "ok": bool(corr.get("ok", False)),
                    "absmax": corr.get("absmax"),
                    "corr": corr,
                    "err": None,
                }
            except Exception as e:
                rec = {
                    "ts": now(),
                    "run": run_id,
                    "impl": name,
                    "tags": getattr(fn, "_tags", {}),
                    "wl": wl,
                    "env": env,
                    "lat_ms": None,
                    "compile_ms": None,
                    "peak_bytes": None,
                    "ok": False,
                    "absmax": None,
                    "corr": {},
                    "err": {"type": type(e).__name__, "msg": str(e)},
                }
            with open(jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
