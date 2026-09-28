# bench/validate.py — contract §C6.  exit code 0 = pass, 1 = fail
import argparse
import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common" / "python"))
import nbio

ap = argparse.ArgumentParser()
ap.add_argument("--test", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--tol", type=float, default=1e-10)
a = ap.parse_args()
_, _, p_ref, v_ref = nbio.read_ic_csv(a.ref)
_, _, p_tst, v_tst = nbio.read_ic_csv(a.test)
if p_ref.shape != p_tst.shape:
    sys.exit(f"FAIL: shape {p_tst.shape} != {p_ref.shape}")
rel_p = np.max(np.linalg.norm(p_tst - p_ref, axis=1) / np.linalg.norm(p_ref, axis=1))
rel_v = np.max(np.linalg.norm(v_tst - v_ref, axis=1) / np.linalg.norm(v_ref, axis=1))
ok = rel_p < a.tol and rel_v < a.tol
print(f"{'PASS' if ok else 'FAIL'}  max rel pos err = {rel_p:.3e}  max rel vel err = {rel_v:.3e}  (tol {a.tol:g})")
sys.exit(0 if ok else 1)
