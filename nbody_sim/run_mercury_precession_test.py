"""Validate the 1PN correction against a real, famous result: the
general-relativistic contribution to Mercury's perihelion precession,
~42.98 arcsec/century.

This is the classic isolated two-body test: an isolated Sun-Mercury
system has a perfectly closed Newtonian ellipse (no precession at all).
Adding the limited 1PN correction (onepn.py, applied to Mercury only,
following Tatekawa 2018's "one dominant mass" simplification) should
make Mercury's orbit precess at very close to the textbook GR rate.

Method: integrate with RK4 (required once the correction is
velocity-dependent, see integrators.py), track Mercury's Laplace-Runge-
Lenz (eccentricity) vector every step, and fit the secular drift of its
in-plane angle to get a numerical precession rate -- then compare
against the closed-form 1PN prediction
    delta_phi_per_orbit = 6 pi G M_sun / (c^2 a (1 - e^2)).
"""
import numpy as np
import matplotlib.pyplot as plt

from constants import G
from orbital_elements import build_solar_system
from simulation import accelerations as newtonian_accelerations
from onepn import onepn_correction, C_AU_PER_DAY
from integrators import rk4_step

YEARS = 20.0
DAYS_PER_YEAR = 365.25
DT = 0.05  # days
ARCSEC_PER_RAD = 180.0 / np.pi * 3600.0
DAYS_PER_CENTURY = 36525.0


def make_accel_fn(use_1pn):
    def accel_fn(pos, vel, masses):
        acc = newtonian_accelerations(pos, masses)
        if use_1pn:
            acc = acc + onepn_correction(pos, vel, masses, G, central_index=0)
        return acc
    return accel_fn


def eccentricity_vector(r_vec, v_vec, mu):
    h = np.cross(r_vec, v_vec)
    return np.cross(v_vec, h) / mu - r_vec / np.linalg.norm(r_vec)


def run(use_1pn):
    names, masses, positions, velocities, colors = build_solar_system(
        include_moon=False, bodies=["Mercury"])
    accel_fn = make_accel_fn(use_1pn)

    mu = G * (masses[0] + masses[1])
    n_steps = int(round(YEARS * DAYS_PER_YEAR / DT))
    record_every = max(1, n_steps // 4000)

    times = []
    thetas_raw = []

    e0 = None
    basis_u = basis_w = None

    for step in range(n_steps + 1):
        if step % record_every == 0:
            r_vec = positions[1] - positions[0]
            v_vec = velocities[1] - velocities[0]
            e_vec = eccentricity_vector(r_vec, v_vec, mu)
            h_vec = np.cross(r_vec, v_vec)
            if basis_u is None:
                basis_u = e_vec / np.linalg.norm(e_vec)
                basis_w = np.cross(h_vec / np.linalg.norm(h_vec), basis_u)
                e0 = np.linalg.norm(e_vec)
            theta = np.arctan2(e_vec @ basis_w, e_vec @ basis_u)
            times.append(step * DT)
            thetas_raw.append(theta)

        positions, velocities = rk4_step(positions, velocities, masses, DT, accel_fn)

    times = np.array(times)
    theta_unwrapped = np.unwrap(thetas_raw)
    # linear fit: theta(t) = slope * t + intercept
    slope, intercept = np.polyfit(times, theta_unwrapped, 1)
    rate_arcsec_per_century = slope * ARCSEC_PER_RAD * DAYS_PER_CENTURY
    return times, theta_unwrapped, rate_arcsec_per_century, e0


def analytic_precession_rate(a, e, M_sun):
    """6*pi*G*M / (c^2 * a * (1-e^2)) per orbit, converted to arcsec/century."""
    mu = G * M_sun
    T_orbit = 2 * np.pi * np.sqrt(a ** 3 / mu)  # days, Kepler's third law
    delta_phi_per_orbit = 6 * np.pi * mu / (C_AU_PER_DAY ** 2 * a * (1 - e ** 2))
    rate_per_day = delta_phi_per_orbit / T_orbit
    return rate_per_day * ARCSEC_PER_RAD * DAYS_PER_CENTURY, T_orbit


def main():
    print(f"Integrating Sun-Mercury for {YEARS:.0f} years, dt = {DT} day, RK4...")

    t_newt, th_newt, rate_newt, e_newt = run(use_1pn=False)
    print(f"  Newtonian only : measured precession rate = {rate_newt:8.3f} arcsec/century "
          f"(expect ~0)")

    t_1pn, th_1pn, rate_1pn, e_1pn = run(use_1pn=True)
    print(f"  With 1PN term  : measured precession rate = {rate_1pn:8.3f} arcsec/century")

    a_mercury = 0.38709927  # AU, semi-major axis
    rate_theory, T_orbit = analytic_precession_rate(a_mercury, e_1pn, 1.0)
    print(f"  Analytic 1PN prediction (6*pi*GM/(c^2 a(1-e^2))): {rate_theory:8.3f} arcsec/century")
    print(f"  (Textbook GR value for Mercury: 42.98 arcsec/century; "
          f"orbital period used: {T_orbit:.3f} days)")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t_newt / DAYS_PER_YEAR, th_newt * ARCSEC_PER_RAD, label="Newtonian only", color="#5b8cff")
    ax.plot(t_1pn / DAYS_PER_YEAR, th_1pn * ARCSEC_PER_RAD, label="Newtonian + 1PN", color="#f2a65a")
    ax.set_xlabel("time (years)")
    ax.set_ylabel("longitude of perihelion, unwrapped (arcsec)")
    ax.set_title("Mercury's perihelion precession: Newtonian vs. 1PN-corrected")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("mercury_precession.png", dpi=150)
    print("Saved mercury_precession.png")


if __name__ == "__main__":
    main()
