"""Test inputs and reference answers for the MPI backend, built only from nbody_sim/.

Why this exists: the team's shared initial conditions (data/ic/) and reference
final states (data/ref/) are produced by Fahad's tools, which may not be on
this branch yet. This module lets the MPI backend be verified on its own:

* find_or_make_ic(n)  -> path of an IC CSV. It uses data/ic/ic_N{n}_s42.csv if it
                         exists, otherwise it writes a TEST IC to backends/mpi/out/ic/.
* reference_final(ic, steps, dt) -> positions/velocities after `steps` velocity-Verlet
                         steps, computed by the ORIGINAL serial integrator
                         nbody_sim.simulation.NBodySimulation (independent code).

Test ICs follow the contract's recipe (Sun + 8 planets + Moon, then main-belt
asteroids with the paper's mass range), but they are for correctness checks
only. Final benchmark numbers should use the team's data/ic files.
"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "nbody_sim"))
sys.path.insert(0, str(HERE))

import contract_io as nbio  # noqa: E402
from constants import KG_PER_MSUN  # noqa: E402
from orbital_elements import _rotate_to_ecliptic, _solve_kepler, build_solar_system  # noqa: E402
from simulation import NBodySimulation  # noqa: E402

TEAM_IC_DIR = ROOT / "data" / "ic"
TEST_IC_DIR = HERE / "out" / "ic"


def _asteroid_state(a, e, inc, node, peri, M, mu):
    """Keplerian elements (angles in RADIANS) -> heliocentric position, velocity."""
    E = _solve_kepler(M, e)
    x_orb = a * (np.cos(E) - e)
    y_orb = a * np.sqrt(1 - e ** 2) * np.sin(E)
    E_dot = np.sqrt(mu / a ** 3) / (1 - e * np.cos(E))
    vx_orb = -a * np.sin(E) * E_dot
    vy_orb = a * np.sqrt(1 - e ** 2) * np.cos(E) * E_dot
    return _rotate_to_ecliptic(x_orb, y_orb, vx_orb, vy_orb, node, inc, peri)


def make_test_ic(n, seed=42):
    """Sun + 8 planets + Moon + (n - 10) main-belt asteroids, barycentric frame."""
    names, m, pos, vel, _ = build_solar_system(include_moon=True)
    k = max(0, n - len(names))
    if k:
        rng = np.random.default_rng(seed)
        a = rng.uniform(2.1, 3.3, k)
        e = np.clip(rng.rayleigh(0.07, k), 0.0, 0.3)
        inc = np.radians(np.abs(rng.normal(0.0, 8.0, k)))
        node, peri, M = rng.uniform(0.0, 2 * np.pi, (3, k))
        am = rng.uniform(1e12, 3e17, k) / KG_PER_MSUN          # Zhu (2020) mass range
        mu = nbio.G * m[0]
        states = [_asteroid_state(a[i], e[i], inc[i], node[i], peri[i], M[i], mu) for i in range(k)]
        names = names + [f"A{i + 1:05d}" for i in range(k)]
        m = np.concatenate([m, am])
        pos = np.vstack([pos, [s[0] for s in states]])
        vel = np.vstack([vel, [s[1] for s in states]])
        total = m.sum()                                        # re-centre on the barycentre
        pos = pos - (m[:, None] * pos).sum(0) / total
        vel = vel - (m[:, None] * vel).sum(0) / total
    return names[:n], m[:n], pos[:n], vel[:n]


def find_or_make_ic(n, seed=42):
    """Return (path, kind) where kind is 'team' (data/ic) or 'test' (generated here)."""
    team = TEAM_IC_DIR / f"ic_N{n}_s{seed}.csv"
    if team.exists():
        return team, "team"
    test = TEST_IC_DIR / f"test_ic_N{n}_s{seed}.csv"
    if not test.exists():
        nbio.write_state_csv(str(test), *make_test_ic(n, seed))
    return test, "test"


def reference_final(ic_path, steps, dt=0.1):
    """Run the original serial integrator on the same IC -> (names, masses, pos, vel)."""
    names, m, pos, vel = nbio.read_ic_csv(str(ic_path))
    sim = NBodySimulation(names, m, pos, vel, softening=0.0)
    for _ in range(steps):
        sim.step(dt)
    return names, m, sim.positions, sim.velocities


def compare_states(pos_test, vel_test, pos_ref, vel_ref):
    """Max relative position / velocity error over all bodies (contract §C6 metric)."""
    rel_p = np.max(np.linalg.norm(pos_test - pos_ref, axis=1) / np.linalg.norm(pos_ref, axis=1))
    rel_v = np.max(np.linalg.norm(vel_test - vel_ref, axis=1) / np.linalg.norm(vel_ref, axis=1))
    return float(rel_p), float(rel_v)
