#!/usr/bin/env python3
"""
common/python/export_ic.py — Exports initial condition CSV files for all N-body backends.
Uses the verified J2000 barycentric solar system from nbody_sim, augmented with
Keplerian asteroid particles for N > 10.
"""

import argparse
import math
import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path to import nbody_sim modules
proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(proj_root))
sys.path.insert(0, str(proj_root / "nbody_sim"))

from nbody_sim.constants import G
from nbody_sim.orbital_elements import build_solar_system


def generate_asteroid_field(num_asteroids, seed=42):
    """
    Generates realistic asteroid bodies in the main asteroid belt (2.1 to 3.3 AU).
    Orbits are approximately circular with low inclination around the solar barycenter.
    """
    rng = np.random.default_rng(seed)
    
    # Semi-major axis between 2.1 and 3.3 AU
    a = rng.uniform(2.1, 3.3, num_asteroids)
    # Low eccentricity
    e = rng.uniform(0.01, 0.15, num_asteroids)
    # Inclination in radians (up to ~15 degrees)
    inc = rng.uniform(0.0, np.radians(15.0), num_asteroids)
    # Longitude of ascending node, argument of perihelion, mean anomaly
    omega_node = rng.uniform(0.0, 2 * np.pi, num_asteroids)
    omega_peri = rng.uniform(0.0, 2 * np.pi, num_asteroids)
    M = rng.uniform(0.0, 2 * np.pi, num_asteroids)

    # Masses: small asteroid mass in solar masses (~1e-12 to 1e-10 Msun)
    masses = rng.uniform(1e-14, 1e-10, num_asteroids)

    # Approximate Keplerian orbital coordinates in ecliptic plane
    # r = a * (1 - e^2) / (1 + e*cos(nu))
    # For small e, r ≈ a, v ≈ sqrt(G * M_sun / a)
    v_circ = np.sqrt(G * 1.0 / a)

    # Orbital angle
    theta = M + omega_peri
    x = a * (np.cos(theta) * np.cos(omega_node) - np.sin(theta) * np.sin(omega_node) * np.cos(inc))
    y = a * (np.cos(theta) * np.sin(omega_node) + np.sin(theta) * np.cos(omega_node) * np.cos(inc))
    z = a * (np.sin(theta) * np.sin(inc))

    vx = -v_circ * (np.sin(theta) * np.cos(omega_node) + np.cos(theta) * np.sin(omega_node) * np.cos(inc))
    vy = -v_circ * (np.sin(theta) * np.sin(omega_node) - np.cos(theta) * np.cos(omega_node) * np.cos(inc))
    vz = v_circ * (np.cos(theta) * np.sin(inc))

    pos = np.column_stack([x, y, z])
    vel = np.column_stack([vx, vy, vz])
    names = [f"asteroid_{i+1}" for i in range(num_asteroids)]

    return names, masses, pos, vel


def export_ic(n, seed=42, out_dir="data/ic"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_file = out_path / f"ic_N{n}_s{seed}.csv"

    # Base solar system
    sol_names, sol_masses, sol_pos, sol_vel, _ = build_solar_system(include_moon=True)
    n_base = len(sol_names)

    if n <= n_base:
        names = sol_names[:n]
        masses = sol_masses[:n]
        pos = sol_pos[:n]
        vel = sol_vel[:n]
    else:
        num_asteroids = n - n_base
        ast_names, ast_masses, ast_pos, ast_vel = generate_asteroid_field(num_asteroids, seed=seed)
        names = sol_names + ast_names
        masses = np.concatenate([sol_masses, ast_masses])
        pos = np.vstack([sol_pos, ast_pos])
        vel = np.vstack([sol_vel, ast_vel])

    with open(csv_file, "w") as f:
        f.write("name,mass,x,y,z,vx,vy,vz\n")
        for i in range(n):
            f.write(f"{names[i]},{masses[i]:.17e},{pos[i,0]:.17e},{pos[i,1]:.17e},{pos[i,2]:.17e},"
                    f"{vel[i,0]:.17e},{vel[i,1]:.17e},{vel[i,2]:.17e}\n")

    print(f"Generated {csv_file} (N={n}, seed={seed})")


def main():
    parser = argparse.ArgumentParser(description="Export Initial Conditions (IC) CSV for N-body")
    parser.add_argument("--n", type=int, default=None, help="Number of bodies to export")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for asteroids")
    parser.add_argument("--out-dir", default="data/ic", help="Output directory")
    parser.add_argument("--all", action="store_true", help="Generate all standard benchmark sizes")
    args = parser.parse_args()

    if args.all:
        for count in [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
            export_ic(count, seed=args.seed, out_dir=args.out_dir)
    elif args.n is not None:
        export_ic(args.n, seed=args.seed, out_dir=args.out_dir)
    else:
        export_ic(100, seed=args.seed, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
