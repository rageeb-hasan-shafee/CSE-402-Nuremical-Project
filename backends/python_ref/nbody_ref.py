#!/usr/bin/env python3
"""
backends/python_ref/nbody_ref.py — Python/NumPy single-CPU reference baseline.
Uses vectorized NumPy NBodySimulation from nbody_sim.
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import sys

proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(proj_root))
sys.path.insert(0, str(proj_root / "nbody_sim"))

from nbody_sim.simulation import NBodySimulation


def load_csv(path):
    names, masses, pos, vel = [], [], [], []
    with open(path, "r") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if first:
                first = False
                if "name" in line:
                    continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 8:
                names.append(parts[0])
                masses.append(float(parts[1]))
                pos.append([float(parts[2]), float(parts[3]), float(parts[4])])
                vel.append([float(parts[5]), float(parts[6]), float(parts[7])])
    return names, np.array(masses), np.array(pos), np.array(vel)


def main():
    parser = argparse.ArgumentParser(description="Python/NumPy N-body baseline")
    parser.add_argument("--input", required=True, help="Input IC CSV file")
    parser.add_argument("--steps", type=int, default=10, help="Number of integration steps")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step size")
    parser.add_argument("--final-state", default="", help="Output final state CSV")
    parser.add_argument("--output-json", default="", help="Output JSON result")
    parser.add_argument("--machine", default="unknown", help="Machine tag")
    args = parser.parse_args()

    names, masses, pos, vel = load_csv(args.input)
    n = len(names)

    t0 = time.perf_counter()
    sim = NBodySimulation(names, masses, pos, vel, softening=0.0)
    setup_time = time.perf_counter() - t0

    loop_t0 = time.perf_counter()
    for _ in range(args.steps):
        sim.step(args.dt)
    loop_time = time.perf_counter() - loop_t0

    if args.final_state:
        out_p = Path(args.final_state)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            f.write("name,mass,x,y,z,vx,vy,vz\n")
            for i in range(n):
                f.write(f"{names[i]},{sim.masses[i]:.17e},{sim.positions[i,0]:.17e},"
                        f"{sim.positions[i,1]:.17e},{sim.positions[i,2]:.17e},"
                        f"{sim.velocities[i,0]:.17e},{sim.velocities[i,1]:.17e},"
                        f"{sim.velocities[i,2]:.17e}\n")

    if args.output_json:
        out_j = Path(args.output_json)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "machine": args.machine,
            "backend": "py-numpy",
            "method": "numpy",
            "variant": "numpy",
            "n": n,
            "p": 1,
            "steps": args.steps,
            "dt": args.dt,
            "timings": {
                "setup": setup_time,
                "loop": loop_time,
                "transfer": 0.0,
                "total": setup_time + loop_time,
                "ms_per_step": (1e3 * loop_time / args.steps) if args.steps > 0 else 0.0,
            }
        }
        with open(out_j, "w") as f:
            json.dump(data, f, indent=2)

    print(f"py-numpy N={n} steps={args.steps} loop={loop_time:.4f}s ({1e3 * loop_time / args.steps:.3f} ms/step)")


if __name__ == "__main__":
    main()
