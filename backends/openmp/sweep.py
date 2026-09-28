"""backends/openmp/sweep.py — Owner: Siam.
Automated benchmark sweep runner for cpp-serial and cpp-openmp.
Contract v1 (§C4, §C5).
"""
import argparse
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXE_DIR = ROOT / "backends" / "openmp" / "build"
SERIAL_EXE = str(EXE_DIR / "nbody_serial.exe") if os.name == "nt" else str(EXE_DIR / "nbody_serial")
OMP_EXE = str(EXE_DIR / "nbody_omp.exe") if os.name == "nt" else str(EXE_DIR / "nbody_omp")

DEFAULT_MACHINE = "siam-i5-1340p"
N_LIST_FULL = [100, 500, 1000, 2000, 5000, 10000]
THREADS_LIST = [1, 2, 4, 8, 12, 16, 24]  # 4 P-cores + 8 E-cores = 16 threads; 24 = oversubscribed
VARIANTS = ["static", "dynamic", "simd", "newton3"]


def ensure_ic(n, seed=42):
    ic_path = ROOT / "data" / "ic" / f"ic_N{n}_s{seed}.csv"
    if not ic_path.exists():
        print(f"Generating missing IC: {ic_path.name}")
        cmd = [sys.executable, str(ROOT / "common" / "python" / "export_ic.py"), "--n", str(n), "--seed", str(seed)]
        subprocess.run(cmd, check=True)
    return str(ic_path)


def run_single(exe, backend, variant, n, workers, steps, dt, machine, rep, out_dir):
    ic_file = ensure_ic(n)
    json_name = f"{backend}_{variant}_N{n}_P{workers}_r{rep}.json"
    json_path = out_dir / json_name

    if json_path.exists():
        return  # Skip already completed runs

    cmd = [
        exe,
        "--input", ic_file,
        "--steps", str(steps),
        "--dt", str(dt),
        "--workers", str(workers),
        "--variant", variant,
        "--machine", machine,
        "--output-json", str(json_path)
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"ERROR: {res.stderr}")
    else:
        print(res.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default=DEFAULT_MACHINE)
    ap.add_argument("--quick", action="store_true", help="Run a fast subset for sanity testing")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--repeats", type=int, default=5, help="Number of timed repeats (excluding r=0 warmup)")
    args = ap.parse_args()

    n_list = [100, 500, 1000] if args.quick else N_LIST_FULL
    threads = [1, 4, 8, 16] if args.quick else THREADS_LIST
    steps = 20 if args.quick else args.steps

    results_base = ROOT / "bench" / "results"
    serial_dir = results_base / "cpp-serial" / args.machine
    omp_dir = results_base / "cpp-openmp" / args.machine
    serial_dir.mkdir(parents=True, exist_ok=True)
    omp_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting OpenMP & Serial Benchmark Sweep on [{args.machine}] ===")

    # 1. cpp-serial baseline sweep
    print("\n--- Running cpp-serial baseline ---")
    for n in n_list:
        for r in range(args.repeats + 1):  # r=0 is warmup, 1..repeats are timed
            run_single(SERIAL_EXE, "cpp-serial", "static", n, 1, steps, args.dt, args.machine, r, serial_dir)

    # 2. cpp-openmp thread scaling sweep (static variant)
    print("\n--- Running cpp-openmp thread scaling (static) ---")
    for n in n_list:
        for p in threads:
            for r in range(args.repeats + 1):
                run_single(OMP_EXE, "cpp-openmp", "static", n, p, steps, args.dt, args.machine, r, omp_dir)

    # 3. cpp-openmp variants comparison (dynamic, simd, newton3)
    print("\n--- Running cpp-openmp variant sweeps (dynamic, simd, newton3) ---")
    for var in ["dynamic", "simd", "newton3"]:
        for n in ([100, 1000, 5000] if not args.quick else [100, 1000]):
            for p in ([1, 4, 16] if not args.quick else [4]):
                for r in range(args.repeats + 1):
                    run_single(OMP_EXE, "cpp-openmp", var, n, p, steps, args.dt, args.machine, r, omp_dir)

    print("\n=== Benchmark Sweep Completed Successfully! ===")


if __name__ == "__main__":
    main()
