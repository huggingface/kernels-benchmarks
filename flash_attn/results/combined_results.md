---
title: "Flash Attention Benchmark - Combined Results"
author: "uvnote"
theme: "dark"
syntax_theme: "monokai"
show_line_numbers: true
collapse_code: false
custom_css: |
    #output-setup {
        overflow-x: auto;
    }
    .cell-output {
        overflow: scroll;
    }
    .cell-stdout {
        width: max-content;
        overflow: scroll;
    }
    .cell-stderr {
        width: max-content;
        overflow: scroll;
        max-height: 300px;
    }
    svg {
        max-width: 100%;
        height: auto;
        cursor: crosshair;
    }

    /* Hover effects for series lines */
    .series path {
        stroke-width: 6 !important; /* make lines easier to hover */
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1));
    }

    /* remove border on focus */
    g:focus {
        outline: none !important;
    }

    .series:hover path {
        stroke-width: 3;
        filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.2));
        /* transform: translateY(-1px); */
    }

    .series:hover circle {
        r: 5;
        filter: drop-shadow(0 2px 6px rgba(0, 0, 0, 0.3));
    }

    /* Individual series hover colors with glow */
    #series--torch-flash-ma:hover path {
        stroke: #0066cc;
        stroke-width: 4;
        filter: drop-shadow(0 0 8px #1f77b4);
    }

    #series--torch-mem-eff:hover path {
        stroke: #ff6600;
        stroke-width: 4;
        filter: drop-shadow(0 0 8px #ff7f0e);
    }

    #series--xformers-meff:hover path {
        stroke: #228833;
        stroke-width: 4;
        filter: drop-shadow(0 0 8px #2ca02c);
    }

    #series--torch-flash-compiled-default:hover path {
        stroke: #cc0000;
        stroke-width: 4;
        filter: drop-shadow(0 0 8px #d62728);
    }

    #series--torch-flash-compiled-max-autotune:hover path {
        stroke: #7733aa;
        stroke-width: 4;
        filter: drop-shadow(0 0 8px #9467bd);
    }

    #series--hf-kernels-flash-attn:hover path {
        stroke: #664422;
        stroke-width: 4;
        filter: drop-shadow(0 0 8px #8c564b);
    }

    #series--hf-kernels-flash-attn3:hover path {
        stroke: #cc3399;
        stroke-width: 4;
        filter: drop-shadow(0 0 8px #e377c2);
    }

    /* Cursor changes */
    .series {
        cursor: pointer;
    }

    .series:hover {
        cursor: pointer;
    }

    /* Tooltip styles */
    .tooltip {
        position: absolute;
        background: rgba(0, 0, 0, 0.9);
        color: white;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 500;
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.3s ease;
        z-index: 1000;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
    }

    .tooltip.show {
        opacity: 1;
    }

    /* Legend hover effects */
    .legend g:hover text {
        font-weight: bold;
        fill: #333;
    }

    .legend g {
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .legend g:hover {
        transform: translateX(5px);
    }

    /* Subtle animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    .series:active path {
        animation: pulse 0.3s ease;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .chart-container {
            padding: 15px;
            margin: 10px;
        }
    }

    /* Loading animation */
    .chart-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        transform: translateX(-100%);
        animation: shimmer 2s ease-in-out;
    }

    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
---

# Flash Attention Benchmarks - Aggregated Results

This document combines benchmark results from multiple attention implementations
using cross-file dependencies.

## Combined Summary and Visualization

![artifact:latency.svg]

![artifact:latency.csv]

```python id=combine collapse-code=true collapse-output=true needs=../impls/flash_attention.md:benchmark,../impls/mem_efficient_attention.md:benchmark,../impls/xformers.md:benchmark,../impls/compiled_variants.md:benchmark_default,../impls/compiled_variants.md:benchmark_max_autotune,../impls/hf_kernels_flash_attn.md:benchmark,../impls/hf_kernels_flash_attn3.md:benchmark outputs=latency.svg,latency.csv
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "torch",
#     "kernels-benchmark-tools",
#     "matplotlib",
# ]
#
# [tool.uv.sources]
# kernels-benchmark-tools = { git = "https://github.com/drbh/kernels-benchmark-tools.git", branch = "main" }
# ///
import os
import sys
from pathlib import Path
import json
import torch  # noqa: F401  # imported because upstream may expect torch to be importable
import kernels_benchmark_tools as kbt

# --- Matplotlib setup and helpers ------------------------------------------------
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv


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

def _tag_current_figure(default_series_prefix="series"):
    """Attach SVG ids (gid) to key artists so they can be targeted from CSS."""
    fig = plt.gcf()
    if fig is None:
        return

    # Tag the figure itself
    fig.set_gid("figure--latency")

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
            # Matplotlib uses labels beginning with "_" for non-legendable items
            label = raw_label if not raw_label.startswith("_") else f"{default_series_prefix}"
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

def _postprocess_svg_add_classes(svg_path: Path):
    """Add convenient CSS classes alongside ids (e.g., class='series grid grid-x')."""
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
                # Preserve any existing class (unlikely from Matplotlib)
                existing = el.attrib.get("class", "")
                el.set("class", (existing + " " + " ".join(cls)).strip())
        tree.write(svg_path, encoding="utf-8", xml_declaration=True)
    except Exception as e:
        print(f"✗ SVG postprocess (classes) skipped: {e}")

# Monkey-patch savefig to force SVG & ensure tagging occurs even if kbt.viz saves internally.
_orig_savefig = plt.savefig
def _savefig_svg(fname, *args, **kwargs):
    # Always save as SVG at a stable path for the artifact system
    out = Path("latency.svg")
    kwargs["format"] = "svg"
    # Ensure everything we care about has ids before export
    _tag_current_figure()
    res = _orig_savefig(out, *args, **kwargs)
    # Add helpful CSS classes on top of ids
    _postprocess_svg_add_classes(out)
    print(f"✓ Combined visualization saved as {out}")
    return res

plt.savefig = _savefig_svg  # apply patch

# Capture close calls in case kbt.viz() closes figures before we re-save
_orig_close = plt.close
_last_closed = {"fig": None}
def _capture_close(arg=None):
    try:
        if hasattr(arg, "savefig"):  # looks like a Figure
            _last_closed["fig"] = arg
        else:
            _last_closed["fig"] = plt.gcf()
    finally:
        return _orig_close(arg)
plt.close = _capture_close

# --- Locate benchmark artifacts --------------------------------------------------
cache_dirs = {
    "Flash (PyTorch SDPA)": os.environ.get('UVNOTE_FILE_FLASH_ATTENTION_BENCHMARK'),
    "MemEff (PyTorch SDPA)": os.environ.get('UVNOTE_FILE_MEM_EFFICIENT_ATTENTION_BENCHMARK'),
    "Flash Attn 2": os.environ.get('UVNOTE_FILE_FLASH_ATTN2_BENCHMARK'),
    "xFormers": os.environ.get('UVNOTE_FILE_XFORMERS_BENCHMARK'),
    "SageAttention": os.environ.get('UVNOTE_FILE_SAGE_ATTENTION_BENCHMARK'),
    "Compiled (default)": os.environ.get('UVNOTE_FILE_COMPILED_VARIANTS_BENCHMARK_DEFAULT'),
    "Compiled (max-autotune)": os.environ.get('UVNOTE_FILE_COMPILED_VARIANTS_BENCHMARK_MAX_AUTOTUNE'),
    "HF Kernels Flash Attn": os.environ.get('UVNOTE_FILE_HF_KERNELS_FLASH_ATTN_BENCHMARK'),
    "HF Kernels Flash Attn3": os.environ.get('UVNOTE_FILE_HF_KERNELS_FLASH_ATTN3_BENCHMARK'),
}

print("LOADING BENCHMARK DATA")
for name, cache_dir in cache_dirs.items():
    print(f"{name:30s}: {cache_dir}")
print()

file_mapping = {
    "Flash (PyTorch SDPA)": "attn.jsonl",
    "MemEff (PyTorch SDPA)": "attn.jsonl",
    "Flash Attn 2": "attn.jsonl",
    "xFormers": "attn.jsonl",
    "SageAttention": "attn.jsonl",
    "Compiled (default)": "attn_default.jsonl",
    "Compiled (max-autotune)": "attn_max_autotune.jsonl",
    "HF Kernels Flash Attn": "attn.jsonl",
    "HF Kernels Flash Attn3": "attn.jsonl",
}

all_paths = []
for name, cache_dir in cache_dirs.items():
    if cache_dir:
        path = Path(cache_dir) / file_mapping[name]
        if path.exists() and path.stat().st_size > 0:
            all_paths.append(str(path))
            print(f"✓ Found {name}: {path}")
        else:
            print(f"⊘ Empty/Missing {name}: {path}")
    else:
        print(f"✗ No cache dir for {name}")
print()

if not all_paths:
    print("ERROR: No benchmark data files found!")
    # restore patched functions before exiting
    plt.savefig = _orig_savefig
    plt.close = _orig_close
    sys.exit(1)

# --- Summary + Visualization -----------------------------------------------------
print("COMBINED BENCHMARK SUMMARY\n")
kbt.summarize(all_paths)
print("\nGENERATING COMBINED VISUALIZATION\n")

try:
    # If kbt.viz saves internally, our patched savefig ensures SVG gets written,
    # and it will carry ids/classes for CSS styling.
    kbt.viz(all_paths)
    # Safety net: if kbt.viz didn't save, save now.
    # if not Path("latency.svg").exists():
    #     _tag_current_figure()
    # plt.savefig("latency.svg")

    plt.savefig("latency.svg")  # ensure saved with tagging

    print("✓ SVG visualization ready: latency.svg!")
except ImportError as e:
    print(f"✗ Visualization requires matplotlib: {e}")
except Exception as e:
    print(f"✗ Visualization failed: {e}")
finally:
    # Clean up patches to avoid side effects in later cells
    plt.savefig = _orig_savefig
    plt.close = _orig_close

print()
print("ANALYSIS COMPLETE")
print(f"Total implementations analyzed: {len(all_paths)}")
print(f"\nImplementations included:")
for name, cache_dir in cache_dirs.items():
    if cache_dir:
        path = Path(cache_dir) / file_mapping[name]
        if path.exists() and path.stat().st_size > 0:
            print(f"  ✓ {name}")



# Collect all benchmark data and export to CSV
all_data = {}
for name, cache_dir in cache_dirs.items():
    if cache_dir:
        path = Path(cache_dir) / file_mapping[name]
        if path.exists() and path.stat().st_size > 0:
            with open(path, 'r') as f:
                records = [json.loads(line) for line in f]
                all_data[name] = records

# Export to CSV
csv_path = Path("latency.csv")
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write header
    header = ["Implementation", "Impl ID", "Workload", "Batch", "Seq Length", "Heads", "Head Dim", "Dtype",
              "Mean (ms)", "P10 (ms)", "P50 (ms)", "P90 (ms)", "Reps", 
            #   "Compile (ms)", 
              "Peak Mem (MB)", "Backend", "Family"]
    writer.writerow(header)

    # Write data rows
    for impl_name, records in all_data.items():
        for record in records:
            wl = record.get('wl', {})
            lat = record.get('lat_ms', {})
            tags = record.get('tags', {})

            row = [
                impl_name,
                record.get('impl', ''),
                wl.get('name', ''),
                wl.get('batch', ''),
                wl.get('seq_len', ''),
                wl.get('heads', ''),
                wl.get('head_dim', ''),
                wl.get('dtype', ''),
                lat.get('mean', ''),
                lat.get('p10', ''),
                lat.get('p50', ''),
                lat.get('p90', ''),
                lat.get('reps', ''),
                # record.get('compile_ms', ''),
                round(record.get('peak_bytes', 0) / 1024 / 1024, 2) if record.get('peak_bytes') else '',
                tags.get('backend', ''),
                tags.get('family', ''),
            ]
            writer.writerow(row)

print(f"✓ CSV export complete: {csv_path}")
print(f"Total implementations: {len(all_data)}")
print(f"Total records: {sum(len(records) for records in all_data.values())}")

```



<script>
// Configuration object mapping series IDs to their URLs and descriptions
const seriesConfig = {
    'series--torch-flash-ma': {
        url: '../impls/flash_attention.html',
        name: 'PyTorch Flash Attention (Math)',
        description: 'PyTorch built-in scaled dot product attention with math backend'
    },
    'series--torch-mem-eff': {
        url: '../impls/mem_efficient_attention.html',
        name: 'PyTorch Memory Efficient',
        description: 'PyTorch memory-efficient attention implementation'
    },
    'series--xformers-meff': {
        url: '../impls/xformers.html',
        name: 'xFormers Memory Efficient',
        description: 'Facebook Research xFormers memory-efficient attention'
    },
    'series--torch-flash-compiled-default': {
        url: '../impls/compiled_variants.html',
        name: 'PyTorch Flash Compiled (Default)',
        description: 'PyTorch compiled Flash Attention with default settings'
    },
    'series--torch-flash-compiled-max-autotune': {
        url: '../impls/compiled_variants.html',
        name: 'PyTorch Flash Compiled (Max Autotune)',
        description: 'PyTorch compiled Flash Attention with maximum auto-tuning'
    },
    'series--hf-kernels-flash-attn': {
        url: '../impls/hf_kernels_flash_attn.html',
        name: 'HuggingFace Flash Attention',
        description: 'HuggingFace kernels implementation of Flash Attention'
    },
    'series--hf-kernels-flash-attn3': {
        url: '../impls/hf_kernels_flash_attn3.html',
        name: 'HuggingFace Flash Attention 3',
        description: 'HuggingFace kernels implementation of Flash Attention 3'
    }
};

// Get tooltip element (create if it doesn't exist)
let tooltip = document.getElementById('tooltip');
if (!tooltip) {
    tooltip = document.createElement('div');
    tooltip.id = 'tooltip';
    tooltip.className = 'tooltip';
    document.body.appendChild(tooltip);
}

// Add event listeners to each series
Object.keys(seriesConfig).forEach(seriesId => {
    const seriesElement = document.getElementById(seriesId);
    const config = seriesConfig[seriesId];
    
    if (seriesElement) {
        // Mouse enter - show tooltip
        seriesElement.addEventListener('mouseenter', (e) => {
            tooltip.innerHTML = `
                <strong>${config.name}</strong><br>
                <span style="font-size: 12px; opacity: 0.9;">${config.description}</span><br>
                <span style="font-size: 11px; opacity: 0.7;">Click to view documentation</span>
            `;
            tooltip.classList.add('show');
        });

        // Mouse move - update tooltip position
        seriesElement.addEventListener('mousemove', (e) => {
            const rect = document.querySelector('svg').getBoundingClientRect();
            tooltip.style.left = (e.clientX + 15) + 'px';
            tooltip.style.top = (e.clientY - 10) + 'px';
        });

        // Mouse leave - hide tooltip
        seriesElement.addEventListener('mouseleave', () => {
            tooltip.classList.remove('show');
        });

        // Click - open URL in new tab
        seriesElement.addEventListener('click', (e) => {
            e.preventDefault();
            window.open(config.url, '_blank', 'noopener,noreferrer');
            
            // Optional: Add visual feedback for click
            seriesElement.style.transform = 'scale(0.98)';
            setTimeout(() => {
                seriesElement.style.transform = '';
            }, 150);
        });

        // Add cursor pointer style
        seriesElement.style.cursor = 'pointer';
    }
});

// Also add click handlers to legend items
const legendLabels = document.querySelectorAll('[id^="legend-label--"]');
legendLabels.forEach(label => {
    const labelId = label.id.replace('legend-label--', 'series--');
    const config = seriesConfig[labelId];
    
    if (config) {
        label.style.cursor = 'pointer';
        
        label.addEventListener('click', (e) => {
            e.preventDefault();
            window.open(config.url, '_blank', 'noopener,noreferrer');
        });

        label.addEventListener('mouseenter', (e) => {
            tooltip.innerHTML = `
                <strong>${config.name}</strong><br>
                <span style="font-size: 12px; opacity: 0.9;">${config.description}</span><br>
                <span style="font-size: 11px; opacity: 0.7;">Click to view documentation</span>
            `;
            tooltip.classList.add('show');
        });

        label.addEventListener('mousemove', (e) => {
            tooltip.style.left = (e.clientX + 15) + 'px';
            tooltip.style.top = (e.clientY - 10) + 'px';
        });

        label.addEventListener('mouseleave', () => {
            tooltip.classList.remove('show');
        });
    }
});

// Keyboard accessibility - Enter key support
document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target.closest('[id^="series--"]')) {
        const seriesId = e.target.closest('[id^="series--"]').id;
        const config = seriesConfig[seriesId];
        if (config) {
            window.open(config.url, '_blank', 'noopener,noreferrer');
        }
    }
});

// Add focus styles for keyboard navigation
const allInteractiveElements = document.querySelectorAll('[id^="series--"], [id^="legend-label--"]');
allInteractiveElements.forEach(element => {
    element.setAttribute('tabindex', '0');
    element.addEventListener('focus', () => {
        element.style.outline = '2px solid #007acc';
        element.style.outlineOffset = '2px';
    });
    element.addEventListener('blur', () => {
        element.style.outline = '';
        element.style.outlineOffset = '';
    });
});
</script>


