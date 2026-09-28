"""Reference backend: py-numpy (default) and py-loop. Contract v1. Owner: Fahad."""
import math
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "nbody_sim"))
sys.path.insert(0, str(ROOT / "common" / "python"))
import nbio  # noqa: E402
from simulation import accelerations  # noqa: E402


def accel_loop(pos, m):
    n = len(m)
    p_ = pos.tolist()
    m_ = m.tolist()
    out = [[0.0, 0.0, 0.0] for _ in range(n)]
    for p in range(n):
        xp, yp, zp = p_[p]
        ax = ay = az = 0.0
        for j in range(n):
            if j == p:
                continue
            dx = p_[j][0] - xp
            dy = p_[j][1] - yp
            dz = p_[j][2] - zp
            r2 = dx * dx + dy * dy + dz * dz
            f = nbio.G * m_[j] / (r2 * math.sqrt(r2))
            ax += f * dx
            ay += f * dy
            az += f * dz
        out[p] = [ax, ay, az]
    return np.array(out)


def main():
    a = nbio.parse_args(default_variant="numpy")
    names, m, pos, vel = nbio.read_ic_csv(a.input)
    force = (lambda r: accelerations(r, m)) if a.variant == "numpy" else (lambda r: accel_loop(r, m))
    backend = "py-numpy" if a.variant == "numpy" else "py-loop"

    t0 = time.perf_counter()
    acc = force(pos)
    t_setup = time.perf_counter() - t0
    dt, t_force = a.dt, 0.0
    t0 = time.perf_counter()
    for _ in range(a.steps):
        pos += vel * dt + 0.5 * acc * dt * dt
        tf = time.perf_counter()
        new = force(pos)
        t_force += time.perf_counter() - tf
        vel += 0.5 * (acc + new) * dt
        acc = new
    t_loop = time.perf_counter() - t0

    if a.final_state:
        nbio.write_state_csv(a.final_state, names, m, pos, vel)
    nbio.write_result_json(a, backend, "python", len(m), 1, t_setup, t_loop, t_force)
    print(f"{backend} N={len(m)} steps={a.steps} loop={t_loop:.4f}s ({t_loop / a.steps * 1e3:.3f} ms/step)")


if __name__ == "__main__":
    main()
