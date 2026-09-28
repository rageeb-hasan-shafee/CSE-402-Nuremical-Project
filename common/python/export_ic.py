"""Generate contract initial conditions (§C2).
    python common/python/export_ic.py --n 1000     -> data/ic/ic_N1000_s42.csv
    python common/python/export_ic.py --all        -> every N in the benchmark matrix
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "nbody_sim"))
sys.path.insert(0, str(ROOT / "common" / "python"))
from orbital_elements import build_solar_system  # noqa: E402
import nbio  # noqa: E402

N_ALL = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]


def export(n, seed=42):
    names, m, pos, vel, _ = build_solar_system(include_moon=True, n_asteroids=max(0, n - 10), seed=seed)
    path = ROOT / "data" / "ic" / f"ic_N{n}_s{seed}.csv"
    nbio.write_state_csv(str(path), names, m, pos, vel)
    print(f"wrote {path.relative_to(ROOT)}  ({len(names)} bodies)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    for n in (N_ALL if a.all else [a.n]):
        export(n, a.seed)
