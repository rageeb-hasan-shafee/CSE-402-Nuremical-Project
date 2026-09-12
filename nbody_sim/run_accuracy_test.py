"""Reproduce the base paper's accuracy test (Section 4, Fig. 2): run the
velocity Verlet integrator over several years for a range of time step
sizes and plot the percentage error in total energy over time.

The paper found the Earth's orbit error stays within about +-0.02% for
dt = 0.1 day over four years, while the Moon's orbit error is roughly 70x
larger for the same dt (finer time resolution needed for tightly-bound,
fast-orbiting bodies). This script produces the same kind of plot for our
Sun+planets(+Moon) model so the two can be compared directly.
"""
import matplotlib.pyplot as plt
import numpy as np

from orbital_elements import build_solar_system
from simulation import NBodySimulation

YEARS = 4.0
DAYS_PER_YEAR = 365.25
T_END = YEARS * DAYS_PER_YEAR
DT_LIST = [0.05, 0.1, 0.5, 1.0]  # days


def run_energy_error(dt):
    names, masses, positions, velocities, colors = build_solar_system(include_moon=True)
    sim = NBodySimulation(names, masses, positions, velocities, colors)
    n_steps = int(round(T_END / dt))
    record_every = max(1, n_steps // 400)
    result = sim.run(n_steps, dt, record_every=record_every)
    e0 = result["energy"][0]
    pct_error = 100.0 * (result["energy"] - e0) / abs(e0)
    return result["t"] / DAYS_PER_YEAR, pct_error


def main():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for dt in DT_LIST:
        years, err = run_energy_error(dt)
        ax.plot(years, err, label=f"dt = {dt} day")

    ax.set_xlabel("time (years)")
    ax.set_ylabel(r"total energy error $\Delta E$ (%)")
    ax.set_title("Velocity Verlet accuracy: energy conservation vs. time step")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = "accuracy_energy_error.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
