"""Serial N-body gravitational simulation using the velocity Verlet
integrator, reproducing the core numerical method of the base paper:

    Tailin Zhu, "N-body Simulations of the Solar System with CPU-based
    Parallel Methods", School of Physics, University of Bristol (2020).

Equation of motion (Newtonian gravity), for body p among N bodies:

    r_ddot_p = sum_{j != p} G m_j (r_j - r_p) / |r_j - r_p|^3

Integrated with the second-order velocity Verlet scheme:

    r_{i+1} = r_i + v_i*dt + 1/2 a_i*dt^2
    v_{i+1} = v_i + 1/2 (a_i + a_{i+1})*dt

This is the serial CPU baseline (project scope step 1); the force
evaluation is isolated in `accelerations()` so it can later be swapped
for OpenMP/MPI/CUDA-parallel implementations without touching the
integrator.
"""
import numpy as np

from constants import G


def accelerations(positions, masses, softening=0.0):
    """Vectorised O(N^2) gravitational acceleration for every body.

    positions : (N, 3) array [AU]
    masses    : (N,) array [Msun]
    softening : Plummer softening length [AU] to avoid singularities on
                close encounters (0 disables it).

    Returns (N, 3) array of accelerations [AU/day^2].
    """
    diff = positions[None, :, :] - positions[:, None, :]  # r_j - r_p, shape (N,N,3)
    dist2 = np.sum(diff ** 2, axis=-1) + softening ** 2
    np.fill_diagonal(dist2, 1.0)  # avoid div-by-zero on the diagonal
    inv_dist3 = dist2 ** -1.5
    np.fill_diagonal(inv_dist3, 0.0)  # a body exerts no force on itself

    acc = G * np.sum(masses[None, :, None] * inv_dist3[:, :, None] * diff, axis=1)
    return acc


def total_energy(positions, velocities, masses):
    """Total mechanical energy of the system (kinetic + potential), used
    as the accuracy diagnostic (energy should be conserved by the true
    solution, per the paper's Section 4 accuracy test)."""
    kinetic = 0.5 * np.sum(masses * np.sum(velocities ** 2, axis=1))

    diff = positions[None, :, :] - positions[:, None, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    potential = -G * np.sum(np.triu(masses[:, None] * masses[None, :] / dist, k=1))

    return kinetic + potential


class NBodySimulation:
    """Velocity-Verlet integrated N-body system."""

    def __init__(self, names, masses, positions, velocities, colors=None, softening=0.0):
        self.names = list(names)
        self.masses = np.asarray(masses, dtype=float)
        self.positions = np.asarray(positions, dtype=float).copy()
        self.velocities = np.asarray(velocities, dtype=float).copy()
        self.colors = colors
        self.softening = softening
        self.acc = accelerations(self.positions, self.masses, self.softening)
        self.t = 0.0

    @property
    def n_bodies(self):
        return len(self.names)

    def step(self, dt):
        """Advance the system by one velocity-Verlet step of size dt [days]."""
        self.positions += self.velocities * dt + 0.5 * self.acc * dt ** 2
        new_acc = accelerations(self.positions, self.masses, self.softening)
        self.velocities += 0.5 * (self.acc + new_acc) * dt
        self.acc = new_acc
        self.t += dt

    def energy(self):
        return total_energy(self.positions, self.velocities, self.masses)

    def run(self, n_steps, dt, record_every=1):
        """Run n_steps of size dt, recording a trajectory snapshot every
        `record_every` steps. Returns dict of time, positions, energy arrays."""
        n_records = n_steps // record_every + 1
        times = np.empty(n_records)
        traj = np.empty((n_records, self.n_bodies, 3))
        energies = np.empty(n_records)

        times[0] = self.t
        traj[0] = self.positions
        energies[0] = self.energy()

        rec = 1
        for step in range(1, n_steps + 1):
            self.step(dt)
            if step % record_every == 0:
                times[rec] = self.t
                traj[rec] = self.positions
                energies[rec] = self.energy()
                rec += 1

        return {"t": times, "positions": traj, "energy": energies, "names": self.names}
