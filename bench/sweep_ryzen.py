"""
bench/sweep_ryzen.py — Automated Benchmark Runner for AMD Ryzen 5 5600G (Fahad's Machine)
Runs all missing CPU benchmarks across:
  1. OpenMP variants (dynamic, simd, newton3, static) across all thread counts & problem sizes
  2. MPI variants (master-worker & allgather) across ranks [1, 2, 4, 6, 12]
  3. Python reference baselines (py-numpy and py-loop)
Skips existing runs automatically to be 100% resumable and idempotent.
"""

import os
import sys
import time
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "bench" / "results"
MACHINE = "fahad-ryzen-5600g"

# Ensure environment variables for clean multithreading & MPI initialization
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

PYTHON_EXE = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not os.path.exists(PYTHON_EXE):
    PYTHON_EXE = sys.executable

SERIAL_EXE = str(ROOT / "backends" / "openmp" / "build" / "nbody_serial.exe")
OMP_EXE = str(ROOT / "backends" / "openmp" / "build" / "nbody_omp.exe")
MPI_SCRIPT = str(ROOT / "backends" / "mpi" / "nbody_mpi.py")
PYREF_SCRIPT = str(ROOT / "backends" / "python_ref" / "nbody_ref.py")


def ensure_ic(n, seed=42):
    ic_path = ROOT / "data" / "ic" / f"ic_N{n}_s{seed}.csv"
    if not ic_path.exists():
        print(f"Generating IC: {ic_path.name}")
        cmd = [
            PYTHON_EXE,
            str(ROOT / "common" / "python" / "export_ic.py"),
            "--n",
            str(n),
            "--seed",
            str(seed),
        ]
        subprocess.run(cmd, check=True)
    return str(ic_path)


def run_openmp_task(variant, n, workers, steps=100, dt=0.1, repeats=3):
    out_dir = RESULTS_DIR / "cpp-openmp" / MACHINE
    out_dir.mkdir(parents=True, exist_ok=True)
    ic_file = ensure_ic(n)

    for r in range(repeats + 1):
        json_path = out_dir / f"cpp-openmp_{variant}_N{n}_P{workers}_r{r}.json"
        if json_path.exists():
            continue

        cmd = [
            OMP_EXE,
            "--input",
            ic_file,
            "--steps",
            str(steps),
            "--dt",
            str(dt),
            "--workers",
            str(workers),
            "--variant",
            variant,
            "--machine",
            MACHINE,
            "--output-json",
            str(json_path),
        ]
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True)
        dur = time.time() - t0
        if res.returncode == 0:
            print(
                f"  [cpp-openmp] {variant.upper():<8} N={n:<5} P={workers:<2} r={r}: {res.stdout.strip()} ({dur:.2f}s)"
            )
        else:
            print(
                f"  [cpp-openmp ERROR] {variant} N={n} P={workers} r={r}: {res.stderr.strip()}"
            )


def run_mpi_task(variant, n, workers, steps=100, dt=0.1, repeats=2):
    out_dir = RESULTS_DIR / "py-mpi" / MACHINE
    out_dir.mkdir(parents=True, exist_ok=True)
    ic_file = ensure_ic(n)

    for r in range(repeats + 1):
        json_path = out_dir / f"py-mpi_{variant}_N{n}_P{workers}_r{r}.json"
        if json_path.exists():
            continue

        # Use mpiexec with .venv python
        cmd = [
            "mpiexec",
            "-n",
            str(workers),
            PYTHON_EXE,
            MPI_SCRIPT,
            "--input",
            ic_file,
            "--steps",
            str(steps),
            "--dt",
            str(dt),
            "--variant",
            variant,
            "--machine",
            MACHINE,
            "--output-json",
            str(json_path),
        ]
        t0 = time.time()
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
        )
        dur = time.time() - t0
        if res.returncode == 0:
            print(
                f"  [py-mpi] {variant.upper():<14} N={n:<5} P={workers:<2} r={r}: {res.stdout.strip()} ({dur:.2f}s)"
            )
        else:
            print(
                f"  [py-mpi ERROR] {variant} N={n} P={workers} r={r}: {res.stderr.strip()}"
            )


def run_python_ref_task(variant, n, steps=100, dt=0.1, repeats=2):
    backend_tag = "py-numpy" if variant == "numpy" else "py-loop"
    out_dir = RESULTS_DIR / backend_tag / MACHINE
    out_dir.mkdir(parents=True, exist_ok=True)
    ic_file = ensure_ic(n)

    for r in range(repeats + 1):
        json_path = out_dir / f"{backend_tag}_{variant}_N{n}_P1_r{r}.json"
        if json_path.exists():
            continue

        cmd = [
            PYTHON_EXE,
            PYREF_SCRIPT,
            "--input",
            ic_file,
            "--steps",
            str(steps),
            "--dt",
            str(dt),
            "--variant",
            variant,
            "--machine",
            MACHINE,
            "--output-json",
            str(json_path),
        ]
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True)
        dur = time.time() - t0
        if res.returncode == 0:
            print(
                f"  [{backend_tag}] {variant.upper():<8} N={n:<5} P=1  r={r}: {res.stdout.strip()} ({dur:.2f}s)"
            )
        else:
            print(
                f"  [{backend_tag} ERROR] {variant} N={n} r={r}: {res.stderr.strip()}"
            )


def main():
    print(f"============================================================")
    print(f" CPU BENCHMARK RUNNER FOR AMD RYZEN 5 5600G (12T)")
    print(f" Machine Target: {MACHINE}")
    print(f"============================================================\n")

    t_start = time.time()

    # 1. MPI master-worker sweep (previously completely missing on Ryzen)
    print(">>> PHASE 1: Running MPI 'master-worker' variant across ranks and N...")
    for n in [100, 500, 1000, 2000, 5000]:
        steps = 100 if n <= 2000 else 50
        for p in [1, 2, 4, 6, 12]:
            run_mpi_task("master-worker", n, p, steps=steps, dt=0.1, repeats=2)

    # 2. MPI allgather for N=10,000 (previously missing high-N point)
    print("\n>>> PHASE 2: Running MPI 'allgather' at N=10,000...")
    for p in [1, 2, 4, 6, 12]:
        run_mpi_task("allgather", 10000, p, steps=25, dt=0.1, repeats=1)

    # 3. OpenMP Variants matrix completion
    print(
        "\n>>> PHASE 3: Completing OpenMP Variants Matrix (simd, dynamic, newton3)..."
    )
    # Missing sizes for 1, 6, 12 threads: N in [500, 2000, 10000]
    for var in ["simd", "dynamic", "newton3"]:
        for n in [500, 2000, 10000]:
            steps = 100 if n < 10000 else 50
            for p in [1, 6, 12]:
                run_openmp_task(var, n, p, steps=steps, dt=0.1, repeats=2)

    # Intermediate thread counts P in [2, 4, 8] across all N
    print("\n>>> PHASE 4: Completing Intermediate OpenMP Thread Counts (P=2, 4, 8)...")
    for var in ["simd", "dynamic", "newton3"]:
        for n in [100, 500, 1000, 2000, 5000]:
            for p in [2, 4, 8]:
                run_openmp_task(var, n, p, steps=100, dt=0.1, repeats=2)

    # 4. Python Reference Baselines
    print("\n>>> PHASE 5: Running Python Reference Baselines (py-numpy & py-loop)...")
    for n in [100, 500, 1000, 2000, 5000]:
        steps = 100 if n <= 1000 else 20
        run_python_ref_task("numpy", n, steps=steps, dt=0.1, repeats=2)

    # Small py-loop for baseline overhead proof
    for n in [100, 500]:
        steps = 20 if n == 100 else 5
        run_python_ref_task("loop", n, steps=steps, dt=0.1, repeats=1)

    total_elapsed = time.time() - t_start
    print(f"\n============================================================")
    print(f" ALL CPU BENCHMARKS COMPLETED IN {total_elapsed:.1f}s!")
    print(f"============================================================\n")


if __name__ == "__main__":
    main()
