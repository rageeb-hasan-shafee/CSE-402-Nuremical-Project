# Orbital Verlet — User Manual

An N-body gravitational simulation of the Solar System, integrated with the
second-order velocity Verlet method. This reproduces the core numerical
method of the base paper:

> Tailin Zhu, *N-body Simulations of the Solar System with CPU-based
> Parallel Methods*, School of Physics, University of Bristol (2020).

The project has two parts that share the same physics (same initial
conditions, same gravitational constant, same integrator):

1. **Python package** (`nbody_sim/`) — the numerical core: builds initial
   conditions from real orbital elements, runs the simulation, and checks
   its own accuracy. This is the serial CPU baseline described in the
   project's Step 1 scope.
2. **Interactive web demo** ("Orbital Verlet") — the same physics
   reimplemented in JavaScript for a live, explorable version with
   sliders, panning/zooming, and a sandbox mode.

---

## 1. Installation

Requires Python 3.9+.

```bash
cd nbody_sim
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. File-by-file reference

### `constants.py`
Shared physical constants. No functions — just values.

| Name | Meaning |
|---|---|
| `GAUSSIAN_K` | Gaussian gravitational constant, `0.01720209895` |
| `G` | Gravitational constant, `= GAUSSIAN_K**2`, in units of AU³·M☉⁻¹·day⁻² |
| `KG_PER_MSUN` | kg per solar mass, for converting real body masses into the simulation's mass unit |
| `AU_IN_KM` | 1 AU in km |

**Why these units?** Working in AU / day / solar masses (instead of SI
metres/seconds/kg) keeps all the numbers in the simulation close to 1,
which avoids floating-point precision problems — the same convention
astronomers and orbital-mechanics software use.

### `orbital_elements.py`
Builds the initial conditions (position + velocity for every body) at
epoch J2000.0, from published Keplerian orbital elements — no internet
or ephemeris file needed.

| Function | What it does |
|---|---|
| `_wrap_deg(angle)` | Wraps an angle into `[0, 360)` degrees. |
| `_solve_kepler(M, e)` | **The numerical method at the heart of this file.** Solves Kepler's equation `E − e·sin(E) = M` for the eccentric anomaly `E`, given mean anomaly `M` and eccentricity `e`, via Newton–Raphson iteration. |
| `_rotate_to_ecliptic(...)` | Rotates a body's 2-D orbital-plane coordinates (and velocity) into the shared 3-D ecliptic J2000 reference frame, using the orbit's inclination, argument of periapsis, and longitude of ascending node. |
| `planet_state_vector(name, T)` | For one named planet, looks up its orbital elements, solves Kepler's equation, and returns its heliocentric position `[AU]` and velocity `[AU/day]` at `T` Julian centuries past J2000.0 (`T=0` for this project). |
| `moon_state_vector_geocentric()` | Same idea, but returns the Moon's position/velocity *relative to Earth* (added to Earth's own vector in `build_solar_system`). |
| `build_solar_system(include_moon, bodies)` | Assembles the full list of bodies (Sun + planets + optionally Moon): names, masses (converted to solar masses), positions, velocities, and display colors. Shifts everything into the **barycentric frame** (zero total momentum) so the whole system doesn't visibly drift during a long integration. |

### `simulation.py`
The integrator itself — this is the part being validated against the
base paper.

| Function / method | What it does |
|---|---|
| `accelerations(positions, masses, softening)` | Computes the gravitational acceleration on every body from every other body at once (vectorized with NumPy, so it's the `O(N²)` force sum the paper describes, without an explicit double loop). `softening` prevents the acceleration from blowing up if two bodies pass very close to each other. |
| `total_energy(positions, velocities, masses)` | Total mechanical energy (kinetic + potential) of the system. The true physical solution conserves this exactly, so tracking it over time is the standard accuracy check for an N-body integrator (this is what the base paper's Fig. 2 plots). |
| `NBodySimulation.step(dt)` | Advances the system by one velocity-Verlet step of size `dt` days: move positions using the current acceleration, recompute acceleration at the new positions, then update velocities using the *average* of the old and new acceleration. |
| `NBodySimulation.energy()` | Convenience wrapper around `total_energy()` for the simulation's current state. |
| `NBodySimulation.run(n_steps, dt, record_every)` | Runs `step()` in a loop for `n_steps`, and every `record_every` steps records the time, all positions, and the total energy — returns these as arrays for plotting/animating. |

### `run_accuracy_test.py`
Reproduces the base paper's accuracy experiment.

| Function | What it does |
|---|---|
| `run_energy_error(dt)` | Builds a fresh solar system, integrates it for 4 simulated years at time step `dt`, and returns the percentage energy drift over time. |
| `main()` | Calls `run_energy_error()` for several `dt` values (0.05 to 1.0 day) and saves a comparison plot, `accuracy_energy_error.png` — the bigger `dt` is, the larger (but still *bounded*, non-drifting) the energy oscillation should be. |

Run it with:
```bash
python run_accuracy_test.py
```

### `visualize.py`
A local interactive viewer built with matplotlib widgets.

| Function | What it does |
|---|---|
| `main()` | Builds the solar system, sets up the plot, sliders, and buttons, then starts a `FuncAnimation` loop. |
| `update(frame)` (inner function) | Called every animation frame: advances the simulation by the number of days set by the "days/frame" slider (using the "dt" slider's step size), redraws body positions and trails, and updates the on-screen energy-drift readout. |
| `on_play`, `on_reset`, `on_check` (inner functions) | Callbacks wired to the Play/Pause button, Reset button, and per-body trail checkboxes. |

Run it with:
```bash
python visualize.py
```
**Controls:**
- **Play / Pause** — toggle the animation.
- **days/frame** slider — how many simulated days pass per drawn frame (simulation speed).
- **dt (days)** slider — the velocity-Verlet integration time step itself (smaller = more accurate, slower).
- **Reset** — rebuild the solar system from its original J2000 initial conditions.
- **Checkboxes** (right side) — show/hide each body's orbit trail.
- Top-left text — elapsed time and the live energy-conservation error (%), the same accuracy metric as `run_accuracy_test.py`.

---

## 3. Interactive web demo ("Orbital Verlet")

The web version runs the identical physics in JavaScript (same initial
J2000 state vectors, same `G`, same velocity-Verlet integrator), rendered
live on a `<canvas>`. No installation needed — just open the published
page.

**Playback panel**
- **Pause / Play** — toggle the simulation.
- **Reset to J2000** — reload the current scenario's original initial conditions.
- **Integration step (dt)** — the velocity-Verlet time step, in days.
- **Simulation speed** — how many simulated days advance per rendered frame.

**Scenario**
- **Full system** — Sun + all 8 planets + Moon.
- **Inner planets** — Sun, Mercury, Venus, Earth, Moon, Mars only (easier to see up close).
- **Sandbox** — just the Sun; add your own bodies (see below).

**Interaction mode**
- **View / pan** — drag empty space to pan the camera, scroll to zoom, click a body (on the canvas or in the list) to lock the camera onto it.
- **Fling a body** — drag on empty space to draw a velocity arrow, then release to launch a new body from the drag's start point with that velocity. The **new body mass** slider sets its mass (in Earth masses) before you throw it.

**Bodies panel** — click any row to track that body with the camera (click again, or click the Sun, to return to the system-wide view); the checkbox shows/hides that body and its trail.

**Accuracy diagnostic** — a live sparkline of the percentage energy drift, the same metric used in the paper and in `run_accuracy_test.py`. It should stay small and oscillating, never drifting steadily in one direction — that's the signature of a correctly implemented velocity-Verlet integrator.

---

## 4. Notes / current scope

- Bodies modeled: Sun, 8 planets, Moon (the base paper also includes
  asteroids and additional minor moons — not yet implemented here).
- The Moon's orbit uses fixed mean elements (no node/perigee precession),
  fine for short demonstration runs but not a precision ephemeris.
- This is the **serial CPU baseline** only. The project's later stages
  (OpenMP, MPI, CUDA parallelization, per the paper and project slide)
  are not yet implemented; `accelerations()` in `simulation.py` is
  isolated specifically so those can be added without changing the
  integrator.
