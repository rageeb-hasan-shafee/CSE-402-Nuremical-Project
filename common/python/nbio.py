"""Shared I/O for Python backends, contract v1 (see common/CONTRACT.md). Owner: Fahad."""
import argparse
import csv
import json
import os

import numpy as np

G = 2.9591220828559115e-4          # AU^3 Msun^-1 day^-2
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
    """-> names (list), masses (N,), positions (N,3), velocities (N,3); all float64, C-contiguous."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if rows[0] != HEADER:
        raise ValueError(f"{path}: bad header {rows[0]}")
    names = [r[0] for r in rows[1:]]
    data = np.array([[float(v) for v in r[1:]] for r in rows[1:]], dtype=np.float64)
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
