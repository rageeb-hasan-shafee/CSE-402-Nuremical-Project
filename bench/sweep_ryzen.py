#!/usr/bin/env python3
"""
bench/sweep_ryzen.py — Automated benchmark runner for Fahad's AMD Ryzen 5 5600G.
Runs cpp-serial and cpp-openmp sweeps adhering strictly to Contract v1.
"""

import argparse
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
MACHINE = "fahad-ryzen-5600g"
BUILD_DIR = ROOT / "backends" / "openmp" / "build"
SERIAL_EXE = BUILD_DIR / "nbody_serial.exe"
OMP_EXE = BUILD_DIR / "nbody_omp.exe"

N_LIST = [100, 500, 1000, 2000, 5000, 10000]
THREADS_STATIC = [1, 2, 4, 6, 8, 12]  # Ryzen 5 5600G has 6 cores / 12 threads
THREADS_VARIANTS = [1, 6, 12]
REPEATS = 5

def build_binaries():
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    cxx = "g++"
    common_inc = str(ROOT / "common" / "cpp")
    src = str(ROOT / "backends" / "openmp" / "nbody_omp.cpp")

    print("[Build] Compiling nbody_serial.exe...")
    cmd_serial = [cxx, "-O3", "-std=c++17", f"-I{common_inc}", src, "-o", str(SERIAL_EXE)]
    subprocess.run(cmd_serial, check=True)

    print("[Build] Compiling nbody_omp.exe...")
    cmd_omp = [cxx, "-O3", "-fopenmp", "-std=c++17", f"-I{common_inc}", src, "-o", str(OMP_EXE)]
    subprocess.run(cmd_omp, check=True)
    print("[Build] Binaries compiled successfully.\n")

def run_command(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        print(f"FAILED: {' '.join(cmd)}\n{p.stderr}")
        return False
    return True

def sweep_serial():
    out_dir = ROOT / "bench" / "results" / "cpp-serial" / MACHINE
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n==========================================")
    print(f" RUNNING SERIAL BENCHMARK (cpp-serial)")
    print(f"==========================================")

    for n in N_LIST:
        ic = ROOT / "data" / "ic" / f"ic_N{n}_s42.csv"
        if not ic.exists():
            print(f"Generating IC for N={n}...")
            subprocess.run([sys.executable, str(ROOT / "common" / "python" / "export_ic.py"), "--n", str(n)], check=True)

        for r in range(REPEATS):
            json_out = out_dir / f"cpp-serial_static_N{n}_P1_r{r}.json"
            if json_out.exists():
                continue
            cmd = [
                str(SERIAL_EXE),
                "--input", str(ic),
                "--steps", "20" if n >= 5000 else "100",
                "--dt", "0.1",
                "--workers", "1",
                "--variant", "static",
                "--machine", MACHINE,
                "--output-json", str(json_out)
            ]
            print(f"  [cpp-serial] N={n} r={r} ...", end=" ", flush=True)
            t0 = time.time()
            if run_command(cmd):
                print(f"done ({time.time() - t0:.2f}s)")

def sweep_openmp():
    out_dir = ROOT / "bench" / "results" / "cpp-openmp" / MACHINE
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n==========================================")
    print(f" RUNNING OPENMP BENCHMARK (cpp-openmp)")
    print(f"==========================================")

    # 1. static sweep across all N and all thread counts
    for n in N_LIST:
        ic = ROOT / "data" / "ic" / f"ic_N{n}_s42.csv"
        for p in THREADS_STATIC:
            for r in range(REPEATS):
                json_out = out_dir / f"cpp-openmp_static_N{n}_P{p}_r{r}.json"
                if json_out.exists():
                    continue
                cmd = [
                    str(OMP_EXE),
                    "--input", str(ic),
                    "--steps", "20" if n >= 5000 else "100",
                    "--dt", "0.1",
                    "--workers", str(p),
                    "--variant", "static",
                    "--machine", MACHINE,
                    "--output-json", str(json_out)
                ]
                print(f"  [omp-static] N={n} P={p} r={r} ...", end=" ", flush=True)
                t0 = time.time()
                if run_command(cmd):
                    print(f"done ({time.time() - t0:.2f}s)")

    # 2. variants sweep (dynamic, simd, newton3) for N=100, 1000, 5000
    for var in ["dynamic", "simd", "newton3"]:
        for n in [100, 1000, 5000]:
            ic = ROOT / "data" / "ic" / f"ic_N{n}_s42.csv"
            for p in THREADS_VARIANTS:
                for r in range(REPEATS):
                    json_out = out_dir / f"cpp-openmp_{var}_N{n}_P{p}_r{r}.json"
                    if json_out.exists():
                        continue
                    cmd = [
                        str(OMP_EXE),
                        "--input", str(ic),
                        "--steps", "20" if n >= 5000 else "100",
                        "--dt", "0.1",
                        "--workers", str(p),
                        "--variant", var,
                        "--machine", MACHINE,
                        "--output-json", str(json_out)
                    ]
                    print(f"  [omp-{var}] N={n} P={p} r={r} ...", end=" ", flush=True)
                    t0 = time.time()
                    if run_command(cmd):
                        print(f"done ({time.time() - t0:.2f}s)")

def sweep_mpi():
    out_dir = ROOT / "bench" / "results" / "py-mpi" / MACHINE
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n==========================================")
    print(f" RUNNING MPI BENCHMARK (py-mpi)")
    print(f"==========================================")

    py_exe = ROOT / ".venv" / "Scripts" / "python.exe"
    if not py_exe.exists():
        py_exe = pathlib.Path(sys.executable)

    import shutil
    mpiexec_bin = shutil.which("mpiexec")
    if not mpiexec_bin or not pathlib.Path(mpiexec_bin).exists():
        for candidate in ["E:/system/scoop/shims/mpiexec.exe", "E:/system/scoop/apps/msmpi/10.1.1/mpiexec.exe", "C:/Program Files/Microsoft MPI/Bin/mpiexec.exe"]:
            if pathlib.Path(candidate).exists():
                mpiexec_bin = candidate
                break
    if not mpiexec_bin:
        mpiexec_bin = "mpiexec"

    # test up to N=5000
    for n in [100, 500, 1000, 2000, 5000]:
        ic = ROOT / "data" / "ic" / f"ic_N{n}_s42.csv"
        for p in [1, 2, 4, 6, 12]:
            for r in range(3):
                json_out = out_dir / f"py-mpi_allgather_N{n}_P{p}_r{r}.json"
                if json_out.exists():
                    continue
                cmd = [
                    str(mpiexec_bin), "-n", str(p),
                    str(py_exe), str(ROOT / "backends" / "mpi" / "nbody_mpi.py"),
                    "--input", str(ic),
                    "--steps", "20" if n >= 5000 else "100",
                    "--dt", "0.1",
                    "--variant", "allgather",
                    "--machine", MACHINE,
                    "--output-json", str(json_out)
                ]
                print(f"  [py-mpi] N={n} P={p} r={r} ...", end=" ", flush=True)
                t0 = time.time()
                if run_command(cmd):
                    print(f"done ({time.time() - t0:.2f}s)")

def main():
    build_binaries()
    sweep_serial()
    sweep_openmp()
    sweep_mpi()
    print("\nAll sweeps completed successfully!")

if __name__ == "__main__":
    main()
