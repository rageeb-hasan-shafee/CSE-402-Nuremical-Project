"""Contract v1 I/O for the MPI backend (see docs/IMPLEMENTATION_GUIDE_ULTRA.md §3).

The team's shared I/O library is meant to live at common/python/nbio.py (owned
by Fahad). Until it exists on this branch, this module provides the same
functions itself, so the MPI backend never has to edit files outside
backends/mpi/. Once common/python/nbio.py is merged, it is used automatically.
"""
import argparse
import csv
import importlib.util
import json
import os
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
_SHARED = ROOT / "common" / "python" / "nbio.py"

G = 2.9591220828559115e-4          # AU^3 Msun^-1 day^-2, == 0.01720209895**2 bit for bit
SCHEMA_VERSION = 1
HEADER = ["name", "mass", "x", "y", "z", "vx", "vy", "vz"]


def parse_args(default_variant="default", argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--variant", default=default_variant)
    ap.add_argument("--machine", default="unknown")
    ap.add_argument("--output-json")
    ap.add_argument("--final-state")
    return ap.parse_args(argv)


def read_ic_csv(path):
    """-> names (list), masses (N,), positions (N,3), velocities (N,3); float64, C-contiguous."""
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    if [h.strip() for h in rows[0]] != HEADER:
        raise ValueError(f"{path}: bad header {rows[0]}")
    names = [r[0] for r in rows[1:]]
    data = np.array([[float(v) for v in r[1:8]] for r in rows[1:]], dtype=np.float64)
    return (names, np.ascontiguousarray(data[:, 0]),
            np.ascontiguousarray(data[:, 1:4]), np.ascontiguousarray(data[:, 4:7]))


def write_state_csv(path, names, masses, pos, vel):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(HEADER)
        for i, nm in enumerate(names):
            w.writerow([nm] + [repr(float(v)) for v in (masses[i], *pos[i], *vel[i])])


def write_result_json(args, backend, lang, n_bodies, workers, t_setup, t_loop,
                      t_force=0.0, t_comm=0.0, t_transfer=0.0, extra=None):
    if not args.output_json:
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    doc = {
        "schema_version": SCHEMA_VERSION, "backend": backend, "variant": args.variant,
        "lang": lang, "n_bodies": int(n_bodies), "workers": int(workers),
        "steps": args.steps, "dt": args.dt,
        "t_setup_s": t_setup, "t_loop_s": t_loop, "t_force_s": t_force,
        "t_comm_s": t_comm, "t_transfer_s": t_transfer,
        "machine": args.machine, "extra": extra or {},
    }
    with open(args.output_json, "w") as f:
        json.dump(doc, f, indent=2)


# Prefer the team's shared library when it has been merged.
if _SHARED.exists():
    _spec = importlib.util.spec_from_file_location("nbio", _SHARED)
    _nbio = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_nbio)
    G = _nbio.G
    parse_args = _nbio.parse_args
    read_ic_csv = _nbio.read_ic_csv
    write_state_csv = _nbio.write_state_csv
    write_result_json = _nbio.write_result_json
    SOURCE = "common/python/nbio.py"
else:
    SOURCE = "backends/mpi/contract_io.py (fallback)"
