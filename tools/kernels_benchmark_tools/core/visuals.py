"""
Visualization utilities for benchmark results.

This module provides reusable components for creating combined benchmark visualizations
with interactive SVG elements, CSS styling, and consistent formatting.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Optional, List
import matplotlib as mpl
import matplotlib.pyplot as plt


def setup_svg_matplotlib():
    """Configure matplotlib for high-quality SVG output with CSS styling support."""
    # Keep text as text (not paths) so CSS can style fonts, size, etc.
    mpl.rcParams["svg.fonttype"] = "none"
    # Make ids deterministic across builds
    mpl.rcParams["svg.hashsalt"] = "latency-benchmark-combined"
    # Avoid auto-closed figures interfering with our tagging
    mpl.rcParams["figure.autolayout"] = True
    # Make background transparent
    mpl.rcParams["figure.facecolor"] = "none"
    mpl.rcParams["axes.facecolor"] = "none"
    mpl.rcParams["savefig.facecolor"] = "none"
    mpl.rcParams["savefig.edgecolor"] = "none"


def _slugify(s: str) -> str:
    """Convert a string to a CSS-friendly slug."""
    s = (s or "").strip().lower()
    keep = []
    for ch in s:
        if ch.isalnum():
            keep.append(ch)
        elif ch in (" ", "-", "_", "/", ".", ":"):
            keep.append("-")
        else:
            keep.append("")
    out = "".join(keep)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-") or "unnamed"


def tag_figure_for_css(
    figure_id: str = "latency", default_series_prefix: str = "series"
):
    """
    Attach SVG ids (gid) to matplotlib figure elements for CSS targeting.

    Args:
        figure_id: ID for the main figure element
        default_series_prefix: Prefix for series that don't have labels
    """
    fig = plt.gcf()
    if fig is None:
        return

    # Tag the figure itself
    fig.set_gid(f"figure--{figure_id}")

    for ax_idx, ax in enumerate(fig.get_axes(), start=1):
        ax.set_gid(f"axes--{ax_idx}")

        # Axis labels & title
        if ax.get_title():
            for t in ax.texts:
                if t.get_text() == ax.get_title():
                    t.set_gid("title--main")
        if ax.xaxis and ax.xaxis.get_label():
            ax.xaxis.label.set_gid("label--x")
        if ax.yaxis and ax.yaxis.get_label():
            ax.yaxis.label.set_gid("label--y")

        # Gridlines
        for i, gl in enumerate(ax.get_xgridlines(), start=1):
            gl.set_gid(f"grid-x--{i}")
        for i, gl in enumerate(ax.get_ygridlines(), start=1):
            gl.set_gid(f"grid-y--{i}")

        # Legend block & entries
        leg = ax.get_legend()
        if leg is not None:
            leg.set_gid("legend")
            for i, txt in enumerate(leg.get_texts(), start=1):
                label_slug = _slugify(txt.get_text())
                txt.set_gid(f"legend-label--{label_slug or i}")

        # Series (lines, patches)
        # Lines
        line_seen = {}
        for ln in getattr(ax, "lines", []):
            raw_label = ln.get_label() or ""
            label = (
                raw_label
                if not raw_label.startswith("_")
                else f"{default_series_prefix}"
            )
            slug = _slugify(label)
            line_seen[slug] = line_seen.get(slug, 0) + 1
            suffix = "" if line_seen[slug] == 1 else f"-{line_seen[slug]}"
            ln.set_gid(f"series--{slug}{suffix}")

        # Patches (bars, areas)
        patch_seen = {}
        for pt in getattr(ax, "patches", []):
            label = getattr(pt, "get_label", lambda: "")() or f"{default_series_prefix}"
            if isinstance(label, str) and label.startswith("_"):
                label = default_series_prefix
            slug = _slugify(label)
            patch_seen[slug] = patch_seen.get(slug, 0) + 1
            suffix = "" if patch_seen[slug] == 1 else f"-{patch_seen[slug]}"
            pt.set_gid(f"series--{slug}{suffix}")


def postprocess_svg_add_classes(svg_path: Path):
    """
    Add CSS classes to SVG elements for easier styling.

    Args:
        svg_path: Path to the SVG file to process
    """
    try:
        import xml.etree.ElementTree as ET

        ET.register_namespace("", "http://www.w3.org/2000/svg")
        tree = ET.parse(svg_path)
        root = tree.getroot()

        for el in root.iter():
            el_id = el.attrib.get("id", "")
            if not el_id:
                continue
            cls = []
            if el_id.startswith("figure--"):
                cls.append("figure")
            elif el_id.startswith("axes--"):
                cls.append("axes")
            elif el_id.startswith("grid-x--"):
                cls += ["grid", "grid-x"]
            elif el_id.startswith("grid-y--"):
                cls += ["grid", "grid-y"]
            elif el_id.startswith("legend"):
                cls.append("legend")
            elif el_id.startswith("label--x"):
                cls.append("xlabel")
            elif el_id.startswith("label--y"):
                cls.append("ylabel")
            elif el_id.startswith("title--"):
                cls.append("title")
            elif el_id.startswith("series--"):
                cls.append("series")
            if cls:
                existing = el.attrib.get("class", "")
                el.set("class", (existing + " " + " ".join(cls)).strip())

        tree.write(svg_path, encoding="utf-8", xml_declaration=True)
    except Exception as e:
        print(f"✗ SVG postprocess (classes) skipped: {e}")


def create_svg_with_tagging(
    output_path: str = "latency.svg", figure_id: str = "latency"
):
    """
    Monkey-patch plt.savefig to ensure SVG output with proper tagging.

    Args:
        output_path: Path to save the SVG file
        figure_id: ID for the figure element

    Returns:
        Tuple of (original_savefig, original_close) for cleanup
    """
    output_file = Path(output_path)
    _orig_savefig = plt.savefig
    _orig_close = plt.close

    def _savefig_svg(fname, *args, **kwargs):
        kwargs["format"] = "svg"
        tag_figure_for_css(figure_id=figure_id)
        res = _orig_savefig(output_file, *args, **kwargs)
        postprocess_svg_add_classes(output_file)
        print(f"✓ Visualization saved as {output_file}")
        return res

    plt.savefig = _savefig_svg

    # Capture close calls in case kbt.viz() closes figures
    _last_closed = {"fig": None}

    def _capture_close(arg=None):
        try:
            if hasattr(arg, "savefig"):
                _last_closed["fig"] = arg
            else:
                _last_closed["fig"] = plt.gcf()
        finally:
            return _orig_close(arg)

    plt.close = _capture_close

    return _orig_savefig, _orig_close


def collect_benchmark_paths(
    cache_env_map: Dict[str, str], output_filename: str = "activation.jsonl"
) -> List[str]:
    """
    Collect benchmark result paths from uvnote environment variables.

    Args:
        cache_env_map: Mapping of display names to environment variable names
                      e.g., {"PyTorch": "UVNOTE_FILE_TORCH_SWIGLU_BENCHMARK"}
        output_filename: Name of the JSONL output file to look for

    Returns:
        List of paths to existing benchmark files
    """
    print("=" * 70)
    print("LOADING BENCHMARK DATA")
    print("=" * 70)
    for name, env_var in cache_env_map.items():
        cache_dir = os.environ.get(env_var)
        status = "✓" if cache_dir else "✗"
        print(f"{status} {name:30s}: {cache_dir or 'NOT SET'}")
    print()

    all_paths = []
    found_count = 0
    skipped_count = 0
    missing_count = 0

    for name, env_var in cache_env_map.items():
        cache_dir = os.environ.get(env_var)
        if cache_dir:
            path = Path(cache_dir) / output_filename
            if path.exists() and path.stat().st_size > 0:
                all_paths.append(str(path))
                print(f"  ✓ Found {name}")
                print(f"     Path: {path}")
                found_count += 1
            else:
                print(f"  ⊘ Skipped {name}")
                print(f"     Path: {path}")
                print(f"     Reason: File not found or empty")
                print(f"     (Benchmark may have exited early - e.g., requires CUDA)")
                skipped_count += 1
        else:
            print(f"  ✗ Missing {name}")
            print(f"     Reason: Environment variable '{env_var}' not set")
            missing_count += 1

    print()
    print("=" * 70)
    print(
        f"Summary: {found_count} found, {skipped_count} skipped, {missing_count} missing"
    )
    print("=" * 70)
    print()

    if skipped_count > 0:
        print("💡 Tip: Some implementations were skipped. Common reasons:")
        print("   • CUDA-only benchmarks running on CPU")
        print("   • Missing dependencies (check stderr output above)")
        print("   • Benchmark explicitly exited (e.g., sys.exit(0))")
        print()

    return all_paths


def generate_combined_results(
    cache_env_map: Dict[str, str],
    output_filename: str = "activation.jsonl",
    svg_filename: str = "latency.svg",
    figure_id: str = "latency",
):
    """
    All-in-one function to generate combined benchmark results with visualization.

    This handles:
    - Setting up matplotlib for SVG output
    - Collecting benchmark paths from environment
    - Generating summary statistics
    - Creating and tagging visualizations
    - Cleanup

    Args:
        cache_env_map: Mapping of display names to environment variable names
        output_filename: Name of benchmark output files (e.g., "activation.jsonl")
        svg_filename: Name of output SVG file
        figure_id: ID for the figure element in SVG

    Example:
        ```python
        from kernels_benchmark_tools.core.visuals import generate_combined_results

        cache_env_map = {
            "PyTorch": "UVNOTE_FILE_TORCH_SWIGLU_BENCHMARK",
            "HF Kernels": "UVNOTE_FILE_HF_KERNELS_SWIGLU_BENCHMARK",
        }

        generate_combined_results(cache_env_map)
        ```
    """
    from .tools import summarize, viz

    # Setup matplotlib
    setup_svg_matplotlib()

    # Collect paths
    all_paths = collect_benchmark_paths(cache_env_map, output_filename)

    if not all_paths:
        print("ERROR: No benchmark data files found!")
        import sys

        sys.exit(1)

    # Setup SVG tagging
    _orig_savefig, _orig_close = create_svg_with_tagging(svg_filename, figure_id)

    try:
        # Generate summary
        print("COMBINED BENCHMARK SUMMARY\n")
        summarize(all_paths)

        # Generate visualization
        print("\nGENERATING COMBINED VISUALIZATION\n")
        try:
            viz(all_paths)
            # Ensure saved with tagging
            plt.savefig(svg_filename)
            print("✓ SVG visualization ready!")
        except ImportError as e:
            print(f"✗ Visualization requires matplotlib: {e}")
        except Exception as e:
            print(f"✗ Visualization failed: {e}")
    finally:
        # Restore original functions
        plt.savefig = _orig_savefig
        plt.close = _orig_close

    # Print summary
    print()
    print("ANALYSIS COMPLETE")
    print(f"Total implementations analyzed: {len(all_paths)}")
    print(f"\nImplementations included:")
    for name, env_var in cache_env_map.items():
        cache_dir = os.environ.get(env_var)
        if cache_dir:
            path = Path(cache_dir) / output_filename
            if path.exists() and path.stat().st_size > 0:
                print(f"  ✓ {name}")
