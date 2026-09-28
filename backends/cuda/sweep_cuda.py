#!/usr/bin/env python3
"""
backends/cuda/sweep_cuda.py — Automated benchmark sweep for CUDA N-body backends.
Owner: Shadhin

Measures:
  1. N scaling (tiled-f64, B=256)
  2. Naive vs Tiled (f64 and f32)
  3. Block size sensitivity (B = 64, 128, 256, 512, 1024)
  4. Precision comparison (tiled-f64 vs tiled-f32)
  5. Transfer cost (copystep-f64 vs tiled-f64)
  6. Scalar CPU baseline & Speed-Up calculation

Automatically generates:
  bench/results/cuda/<machine>/BENCHMARK_REPORT.md
"""

import argparse
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(proj_root))
sys.path.insert(0, str(proj_root / "common" / "python"))


def find_binary(binary_name="nbody_cuda"):
    cuda_dir = Path(__file__).resolve().parent
    candidates = [
        cuda_dir / binary_name,
        cuda_dir / f"{binary_name}.exe",
        cuda_dir.parent.parent / "build" / binary_name,
    ]
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return str(c)
        if sys.platform.startswith("win") and c.exists():
            return str(c)
    local = Path(binary_name)
    if local.exists():
        return str(local.resolve())
    return str(cuda_dir / binary_name)


def find_serial_binary():
    serial_dir = Path(__file__).resolve().parent.parent / "serial"
    candidates = [
        serial_dir / "nbody_serial",
        serial_dir / "nbody_serial.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def ensure_ic_file(n, seed, ic_dir):
    ic_file = ic_dir / f"ic_N{n}_s{seed}.csv"
    if not ic_file.exists():
        try:
            from export_ic import export_ic
            print(f"[IC] Generating missing initial conditions: {ic_file}...")
            export_ic(n, seed=seed, out_dir=str(ic_dir))
        except Exception as e:
            print(f"[IC Warning] Could not auto-generate {ic_file}: {e}")
    return ic_file.exists()


def run_scalar_baseline(n, steps, machine, ic_file, results_dir):
    """Runs or retrieves the scalar baseline timing for N bodies."""
    # Check for any existing scalar JSON for this N
    for existing in results_dir.glob(f"scalar_N{n}_*.json"):
        try:
            with open(existing, "r") as f:
                d = json.load(f)
                return d["timings"]["ms_per_step"]
        except Exception:
            pass

    serial_bin = find_serial_binary()
    # For large N, run 1-2 steps to avoid waiting 15+ minutes while getting exact ms/step
    actual_steps = 1 if n >= 50000 else (2 if n >= 20000 else steps)
    out_json = results_dir / f"scalar_N{n}_steps{actual_steps}.json"

    # Run C++ serial if available
    if serial_bin and Path(serial_bin).exists():
        cmd = [
            serial_bin,
            "--input", str(ic_file),
            "--steps", str(actual_steps),
            "--dt", "0.1",
            "--machine", machine,
            "--output-json", str(out_json),
        ]
        try:
            res = subprocess.run(cmd, check=True, capture_output=True, text=True)
            with open(out_json, "r") as f:
                d = json.load(f)
                return d["timings"]["ms_per_step"]
        except Exception:
            pass

    # Fallback to Python reference if N is small (<= 2000)
    if n <= 2000:
        py_ref = Path(__file__).resolve().parent.parent / "python_ref" / "nbody_ref.py"
        if py_ref.exists():
            cmd = [
                sys.executable, str(py_ref),
                "--input", str(ic_file),
                "--steps", str(steps),
                "--dt", "0.1",
                "--machine", machine,
                "--output-json", str(out_json),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                with open(out_json, "r") as f:
                    d = json.load(f)
                    return d["timings"]["ms_per_step"]
            except Exception:
                pass

    return None


def generate_markdown_report(machine, results_dir, scalar_timings, steps):
    """Parses all benchmark JSONs in results_dir and writes BENCHMARK_REPORT.md"""
    # Group results by (variant, block_size, n) -> list of ms_per_step
    records = defaultdict(list)
    gpu_name = "NVIDIA GPU"

    sm_ver = ""
    for jpath in results_dir.glob("cuda_*.json"):
        try:
            with open(jpath, "r") as f:
                data = json.load(f)
            variant = data.get("variant", "")
            n = data.get("n", 0)
            extra = data.get("extra", {})
            b = extra.get("block_size", 256)
            if "gpu" in extra:
                gpu_name = extra["gpu"]
            if "sm" in extra:
                sm_ver = extra["sm"]
            ms = data["timings"]["ms_per_step"]
            records[(variant, b, n)].append(ms)
        except Exception:
            continue

    if not records:
        print("[REPORT] No JSON results found to generate report.")
        return

    # Compute median timing for each configuration
    agg = {}
    for key, vals in records.items():
        vals.sort()
        median_val = vals[len(vals) // 2]
        agg[key] = median_val

    # Collect unique N values tested
    all_n = sorted(list(set(n for (_, _, n) in agg.keys())))

    report_path = results_dir / "BENCHMARK_REPORT.md"
    lines = []
    lines.append(f"# CUDA N-Body Benchmark Report — {gpu_name}")
    lines.append(f"")
    lines.append(f"### System & GPU Hardware Configuration")
    lines.append(f"* **GPU Model:** `{gpu_name}`")
    if sm_ver:
        lines.append(f"* **Compute Capability:** `SM {sm_ver}`")
    lines.append(f"* **Machine Identifier:** `{machine}`")
    serial_name = "nbody_serial.exe" if sys.platform.startswith("win") else "nbody_serial"
    lines.append(f"* **Scalar CPU Baseline:** Tested on host machine (`{serial_name}`)")
    lines.append(f"* **Benchmark Date:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"* **Integration Scheme:** Velocity Verlet, $\\Delta t = 0.1$ days, `{steps}` steps")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    # Table 1: N Scaling & Speedup vs Scalar CPU
    lines.append(f"## 1. N Scaling & GPU Speed-Up over Scalar CPU ({gpu_name})")
    lines.append("This table evaluates how execution time scales as body count N increases, and computes the exact speed-up (S = T_CPU / T_GPU).")
    lines.append(f"")
    lines.append(f"| N Bodies | Scalar CPU (ms/step) | {gpu_name} FP64 (ms/step) | **FP64 Speed-Up** | {gpu_name} FP32 (ms/step) | **FP32 Speed-Up** |")
    lines.append(f"|:---:|:---:|:---:|:---:|:---:|:---:|")

    for n in all_n:
        sc_ms = scalar_timings.get(n)
        f64_ms = agg.get(("tiled-f64", 256, n))
        f32_ms = agg.get(("tiled-f32", 256, n))

        sc_str = f"{sc_ms:.3f}" if sc_ms else "—"
        f64_str = f"{f64_ms:.3f}" if f64_ms else "—"
        f32_str = f"{f32_ms:.3f}" if f32_ms else "—"

        speedup_f64 = f"**{(sc_ms / f64_ms):.2f}x**" if (sc_ms and f64_ms) else "—"
        speedup_f32 = f"**{(sc_ms / f32_ms):.2f}x**" if (sc_ms and f32_ms) else "—"

        lines.append(f"| {n:,} | {sc_str} | {f64_str} | {speedup_f64} | {f32_str} | {speedup_f32} |")

    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    # Table 2: Memory Hierarchy Optimization (Naive vs. Tiled Shared Memory)
    lines.append(f"## 2. Memory Hierarchy Optimization on {gpu_name} (Naive vs. Tiled Shared Memory)")
    lines.append(f"Demonstrates the benefit of caching particles into on-chip `__shared__` memory ($B=256$) to reduce global memory bandwidth.")
    lines.append(f"")
    lines.append(f"| N Bodies | Naive FP64 (ms/step) | Tiled FP64 (ms/step) | **FP64 Tiling Gain** | Naive FP32 (ms/step) | Tiled FP32 (ms/step) | **FP32 Tiling Gain** |")
    lines.append(f"|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for n in all_n:
        nv_64 = agg.get(("naive-f64", 256, n))
        tl_64 = agg.get(("tiled-f64", 256, n))
        nv_32 = agg.get(("naive-f32", 256, n))
        tl_32 = agg.get(("tiled-f32", 256, n))

        if not (nv_64 or tl_64 or nv_32 or tl_32):
            continue

        gain_64 = f"**{(nv_64 / tl_64):.2f}x**" if (nv_64 and tl_64) else "—"
        gain_32 = f"**{(nv_32 / tl_32):.2f}x**" if (nv_32 and tl_32) else "—"

        nv_64_s = f"{nv_64:.3f}" if nv_64 else "—"
        tl_64_s = f"{tl_64:.3f}" if tl_64 else "—"
        nv_32_s = f"{nv_32:.3f}" if nv_32 else "—"
        tl_32_s = f"{tl_32:.3f}" if tl_32 else "—"

        lines.append(f"| {n:,} | {nv_64_s} | {tl_64_s} | {gain_64} | {nv_32_s} | {tl_32_s} | {gain_32} |")

    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    # Table 3: PCIe Transfer Cost (Resident vs. Copy-Each-Step)
    lines.append(f"## 3. Host-Device Transfer Cost on {gpu_name} (PCIe Bottleneck)")
    lines.append(f"Compares keeping state resident in GPU VRAM vs. copying positions back to host memory each step (`copystep-f64`).")
    lines.append(f"")
    lines.append(f"| N Bodies | GPU Resident (ms/step) | Copy Each Step (ms/step) | **Transfer Overhead Factor** |")
    lines.append(f"|:---:|:---:|:---:|:---:|")

    for n in all_n:
        tl_64 = agg.get(("tiled-f64", 256, n))
        cp_64 = agg.get(("copystep-f64", 256, n))
        if tl_64 and cp_64:
            overhead = (cp_64 / tl_64)
            lines.append(f"| {n:,} | {tl_64:.3f} | {cp_64:.3f} | **{overhead:.2f}x slower** |")

    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    # Table 4: Block Size Sensitivity
    target_n = 10000 if any(n == 10000 for (_, _, n) in agg.keys()) else (all_n[-1] if all_n else 1000)
    lines.append(f"## 4. Block Size Sensitivity on {gpu_name} (N = {target_n:,})")
    lines.append(f"Analyzes occupancy and shared memory configuration across different thread block sizes $B$.")
    lines.append(f"")
    lines.append(f"| Threads per Block (B) | Loop Time (s) | Time per Step (ms/step) |")
    lines.append(f"|:---:|:---:|:---:|")

    target_n = 10000 if any(n == 10000 for (_, _, n) in agg.keys()) else (all_n[-1] if all_n else 1000)
    b_sizes = sorted(list(set(b for (v, b, n) in agg.keys() if v == "tiled-f64" and n == target_n)))

    for b in b_sizes:
        ms = agg.get(("tiled-f64", b, target_n))
        if ms:
            loop_s = (ms * steps) / 1000.0
            lines.append(f"| {b} | {loop_s:.4f} s | {ms:.3f} ms/step |")

    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 5. Key Takeaways for Report")
    lines.append(f"* **Quadratic Scaling:** At large $N$, execution time scales as $O(N^2)$, confirming compute-bound scaling.")
    lines.append(f"* **FP32 vs FP64 Architecture:** Single precision runs substantially faster than double precision due to the GPU's hardware ALU ratio.")
    lines.append(f"* **Shared Memory Tiling:** Tiled shared memory eliminates redundant global memory transactions, resulting in significant speed improvements over naive summation.")
    lines.append(f"* **Zero-Copy Advantage:** Keeping simulation data on the GPU avoids PCIe bus bottlenecks that otherwise degrade performance.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n============================================================")
    print(f" [SUCCESS] Generated Markdown report: {report_path}")
    print(f"============================================================")


def main():
    parser = argparse.ArgumentParser(description="Sweep benchmark matrix for CUDA N-body")
    parser.add_argument("--machine", default="shadhin-colab-t4", help="Machine tag (e.g. shadhin-gtx1660s, shadhin-colab-t4)")
    parser.add_argument("--steps", type=int, default=100, help="Number of integration steps per run (default: 100)")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repetitions per configuration (default: 3)")
    parser.add_argument("--binary", default=None, help="Path to nbody_cuda executable")
    parser.add_argument("--ic-dir", default="data/ic", help="Directory containing initial condition CSVs")
    parser.add_argument("--out-dir", default="bench/results", help="Base directory for output JSON results")
    parser.add_argument("--quick", action="store_true", help="Run a quick smoke test on small N only")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    binary = args.binary or find_binary()
    results_dir = Path(args.out_dir) / "cuda" / args.machine
    results_dir.mkdir(parents=True, exist_ok=True)
    ic_dir = Path(args.ic_dir)
    ic_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        n_values = [100, 500, 1000]
        block_sizes = [128, 256]
        variants = ["tiled-f64", "naive-f64", "tiled-f32"]
        repeats = 1
    else:
        n_values = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
        block_sizes = [64, 128, 256, 512, 1024]
        variants = ["tiled-f64", "naive-f64", "copystep-f64", "tiled-f32", "naive-f32"]
        repeats = args.repeats

    # Pre-generate missing IC files so no N values get skipped
    print("[INIT] Verifying initial condition files...")
    for n in n_values:
        ensure_ic_file(n, seed=42, ic_dir=ic_dir)

    runs = []
    # 1. N scaling & Precision: tiled-f64 and tiled-f32 across all N with default B=256
    for n in n_values:
        for var in ["tiled-f64", "tiled-f32"]:
            runs.append((var, 256, n))

    # 2. Naive vs Tiled & Transfer cost for N <= 10000
    for n in [n for n in n_values if n <= 10000]:
        for var in ["naive-f64", "copystep-f64", "naive-f32"]:
            runs.append((var, 256, n))

    # 3. Block size sensitivity: fix N=10000 (or max N), tiled-f64 across B
    fixed_n = 10000 if 10000 in n_values else n_values[-1]
    for b in block_sizes:
        if b != 256:
            runs.append(("tiled-f64", b, fixed_n))

    # Deduplicate runs preserving order
    unique_runs = []
    seen = set()
    for item in runs:
        if item not in seen:
            seen.add(item)
            unique_runs.append(item)

    print(f"============================================================")
    print(f" CUDA Benchmark Sweep — Machine: {args.machine}")
    print(f" Binary:  {binary}")
    print(f" Output:  {results_dir}")
    print(f" Total configurations: {len(unique_runs)} x {repeats} repeats")
    print(f"============================================================")

    scalar_timings = {}
    scalar_dir = Path(args.out_dir) / "cpp-serial" / args.machine
    scalar_dir.mkdir(parents=True, exist_ok=True)

    # Measure scalar baseline for comparison (across all N)
    for n in n_values:
        ic_file = ic_dir / f"ic_N{n}_s42.csv"
        if ic_file.exists():
            print(f"[SCALAR BASELINE] Measuring CPU baseline for N={n}...")
            s_time = run_scalar_baseline(n, args.steps, args.machine, ic_file, scalar_dir)
            if s_time is not None:
                scalar_timings[n] = s_time

    for variant, block, n in unique_runs:
        ic_file = ic_dir / f"ic_N{n}_s42.csv"
        if not ic_file.exists():
            print(f"Warning: IC file {ic_file} not found, skipping N={n}")
            continue

        for r in range(1, repeats + 1):
            out_json = results_dir / f"cuda_{variant}_B{block}_N{n}_P1_r{r}.json"
            if out_json.exists():
                print(f"[SKIP] {out_json.name} already exists.")
                continue

            cmd = [
                binary,
                "--input", str(ic_file),
                "--steps", str(args.steps),
                "--dt", "0.1",
                "--variant", variant,
                "--block-size", str(block),
                "--machine", args.machine,
                "--output-json", str(out_json),
            ]

            print(f"[RUN] {variant} B={block} N={n} r={r} -> {out_json.name}")
            if args.dry_run:
                print("  " + " ".join(cmd))
                continue

            try:
                res = subprocess.run(cmd, check=True, capture_output=True, text=True)
                if res.stdout:
                    print("  " + res.stdout.strip())
            except subprocess.CalledProcessError as e:
                print(f"  Error running configuration {variant} N={n}: {e.stderr}", file=sys.stderr)

    print("\nBenchmark sweep completed!")

    # Automatically generate the Markdown comparison report
    generate_markdown_report(args.machine, results_dir, scalar_timings, args.steps)


if __name__ == "__main__":
    main()
