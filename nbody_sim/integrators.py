"""Generic RK4 integrator for the N-body equations of motion.

velocity Verlet (`simulation.NBodySimulation`) is symplectic because the
Newtonian Hamiltonian splits cleanly into separate position and momentum
terms. The 1PN correction in `onepn.py` depends on velocity as well as
position, so that split no longer holds -- exactly why Tatekawa (2018)
switches to a classical 4th-order Runge-Kutta integrator for the 1PN
case (Section 4), at the cost of losing Verlet's built-in bound on
long-term energy error.
"""
import numpy as np


def rk4_step(positions, velocities, masses, dt, accel_fn):
    """One RK4 step for dy/dt = [v, a(x, v)].

    accel_fn(positions, velocities, masses) -> (N, 3) accelerations.
    Returns (new_positions, new_velocities).
    """
    def deriv(pos, vel):
        return vel, accel_fn(pos, vel, masses)

    k1v, k1a = deriv(positions, velocities)
    k2v, k2a = deriv(positions + 0.5 * dt * k1v, velocities + 0.5 * dt * k1a)
    k3v, k3a = deriv(positions + 0.5 * dt * k2v, velocities + 0.5 * dt * k2a)
    k4v, k4a = deriv(positions + dt * k3v, velocities + dt * k3a)

    new_pos = positions + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    new_vel = velocities + (dt / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a)
    return new_pos, new_vel
