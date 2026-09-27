#!/usr/bin/env python3
"""
bench/compare.py — Compare CUDA GPU performance and numerical accuracy against scalar CPU baseline.

Usage:
  # Compare benchmark timings & calculate speedup:
  python bench/compare.py --serial-json bench/results/cpp-serial/serial.json --gpu-json bench/results/cuda/cuda.json

  # Compare numerical accuracy between two final state CSVs:
  python bench/compare.py --serial-csv out/serial_state.csv --gpu-csv out/gpu_state.csv --tol 1e-9
"""

import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np


def load_state_csv(path):
    coords = []
    masses = []
    names = []
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
                # x, y, z, vx, vy, vz
                coords.append([float(parts[2]), float(parts[3]), float(parts[4]),
                               float(parts[5]), float(parts[6]), float(parts[7])])
    return names, np.array(masses), np.array(coords)


def compare_accuracy(serial_csv, gpu_csv, tol=1e-9):
    print(f"\n========================================================")
    print(f" NUMERICAL ACCURACY VERIFICATION")
    print(f" Reference (Scalar CPU): {serial_csv}")
    print(f" Test (CUDA GPU):        {gpu_csv}")
    print(f" Tolerance:              {tol:.1e}")
    print(f"========================================================")

    names_s, m_s, data_s = load_state_csv(serial_csv)
    names_g, m_g, data_g = load_state_csv(gpu_csv)

    if len(names_s) != len(names_g):
        print(f"FAIL: Body count mismatch: Serial={len(names_s)}, GPU={len(names_g)}")
        return False

    pos_s = data_s[:, :3]
    pos_g = data_g[:, :3]
    vel_s = data_s[:, 3:]
    vel_g = data_g[:, 3:]

    # Position differences
    pos_diff = np.linalg.norm(pos_g - pos_s, axis=1)
    max_pos_diff = np.max(pos_diff)
    rms_pos_diff = np.sqrt(np.mean(pos_diff**2))

    # Velocity differences
    vel_diff = np.linalg.norm(vel_g - vel_s, axis=1)
    max_vel_diff = np.max(vel_diff)
    rms_vel_diff = np.sqrt(np.mean(vel_diff**2))

    print(f" Max Position Error: {max_pos_diff:.6e} AU")
    print(f" RMS Position Error: {rms_pos_diff:.6e} AU")
    print(f" Max Velocity Error: {max_vel_diff:.6e} AU/day")
    print(f" RMS Velocity Error: {rms_vel_diff:.6e} AU/day")

    if max_pos_diff <= tol:
        print(f"\n >>> RESULT: PASS (All states match within tolerance {tol:.1e}) <<<\n")
        return True
    else:
        print(f"\n >>> RESULT: FAIL (Max position error {max_pos_diff:.2e} exceeds {tol:.1e}) <<<\n")
        return False


def compare_performance(serial_json, gpu_json):
    with open(serial_json, "r") as f:
        s_data = json.load(f)
    with open(gpu_json, "r") as f:
        g_data = json.load(f)

    n = s_data.get("n", g_data.get("n", "unknown"))
    steps = s_data.get("steps", g_data.get("steps", "unknown"))

    t_s = s_data["timings"]["loop"]
    t_g = g_data["timings"]["loop"]
    ms_s = s_data["timings"]["ms_per_step"]
    ms_g = g_data["timings"]["ms_per_step"]

    speedup = t_s / t_g if t_g > 0 else 0.0

    print(f"\n========================================================")
    print(f" PERFORMANCE BENCHMARK COMPARISON (N = {n}, Steps = {steps})")
    print(f"========================================================")
    print(f" Scalar CPU ({s_data.get('backend', 'serial')}):")
    print(f"   Loop Time:     {t_s:.4f} s")
    print(f"   Time per Step: {ms_s:.3f} ms/step")
    print(f" CUDA GPU ({g_data.get('variant', 'cuda')}):")
    print(f"   Loop Time:     {t_g:.4f} s")
    print(f"   Time per Step: {ms_g:.3f} ms/step")
    if "transfer" in g_data["timings"]:
        print(f"   Transfer Time: {g_data['timings']['transfer']:.4f} s")
    print(f"--------------------------------------------------------")
    print(f" >>> GPU SPEED-UP OVER SCALAR CPU: {speedup:.2f}x <<<")
    print(f"========================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Compare CUDA GPU against Scalar CPU baseline")
    parser.add_argument("--serial-json", help="Path to CPU scalar benchmark JSON")
    parser.add_argument("--gpu-json", help="Path to CUDA GPU benchmark JSON")
    parser.add_argument("--serial-csv", help="Path to CPU scalar final state CSV")
    parser.add_argument("--gpu-csv", help="Path to CUDA GPU final state CSV")
    parser.add_argument("--tol", type=float, default=1e-9, help="Accuracy tolerance (default: 1e-9 for f64)")
    args = parser.parse_args()

    if args.serial_csv and args.gpu_csv:
        compare_accuracy(args.serial_csv, args.gpu_csv, tol=args.tol)

    if args.serial_json and args.gpu_json:
        compare_performance(args.serial_json, args.gpu_json)

    if not (args.serial_csv or args.serial_json):
        parser.print_help()


if __name__ == "__main__":
    main()
