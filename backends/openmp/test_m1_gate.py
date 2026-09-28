import subprocess
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXE_DIR = ROOT / "backends" / "openmp" / "build"

SERIAL_EXE = str(EXE_DIR / "nbody_serial.exe")
OMP_EXE = str(EXE_DIR / "nbody_omp.exe")

tests = [
    ("nbody_serial", SERIAL_EXE, "static", 100, 1),
    ("nbody_serial", SERIAL_EXE, "static", 1000, 1),
    ("nbody_omp", OMP_EXE, "static", 100, 4),
    ("nbody_omp", OMP_EXE, "static", 1000, 4),
    ("nbody_omp", OMP_EXE, "dynamic", 100, 4),
    ("nbody_omp", OMP_EXE, "dynamic", 1000, 4),
    ("nbody_omp", OMP_EXE, "simd", 100, 4),
    ("nbody_omp", OMP_EXE, "simd", 1000, 4),
    ("nbody_omp", OMP_EXE, "newton3", 100, 4),
    ("nbody_omp", OMP_EXE, "newton3", 1000, 4),
]

def ensure_data(n):
    ic_path = ROOT / "data" / "ic" / f"ic_N{n}_s42.csv"
    ref_path = ROOT / "data" / "ref" / f"final_N{n}_steps10.csv"
    if not ic_path.exists():
        ic_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Auto-generate] Creating missing IC: {ic_path.name}")
        subprocess.run([sys.executable, str(ROOT / "common" / "python" / "export_ic.py"), "--n", str(n)], check=True)
    if not ref_path.exists():
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Auto-generate] Creating missing Ref: {ref_path.name}")
        subprocess.run([sys.executable, str(ROOT / "backends" / "python_ref" / "nbody_ref.py"),
                        "--input", str(ic_path), "--steps", "10", "--final-state", str(ref_path)], check=True)
    return str(ic_path), str(ref_path)

all_passed = True
for name, exe, var, n, workers in tests:
    ic, ref = ensure_data(n)
    final = str(EXE_DIR / f"final_{name}_{var}_N{n}.csv")
    json_out = str(EXE_DIR / f"test_{name}_{var}_N{n}.json")

    cmd = [
        exe, "--input", ic, "--steps", "10", "--workers", str(workers),
        "--variant", var, "--machine", "siam-i5-1340p",
        "--final-state", final, "--output-json", json_out
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"FAILED RUN: {' '.join(cmd)}\n{res.stderr}")
        all_passed = False
        continue

    val_cmd = [sys.executable, str(ROOT / "bench" / "validate.py"), "--test", final, "--ref", ref]
    val_res = subprocess.run(val_cmd, capture_output=True, text=True)
    val_out = val_res.stdout.strip()
    status = "PASS" if "PASS" in val_out else "FAIL"
    print(f"[{status}] {name:12s} variant={var:8s} N={n:4d} P={workers:2d} -> {val_out}")
    if status != "PASS":
        all_passed = False

if all_passed:
    print("\nALL M1 CORRECTNESS GATE CHECKS PASSED PERFECTLY!")
else:
    print("\nSOME CHECKS FAILED.")
    sys.exit(1)
