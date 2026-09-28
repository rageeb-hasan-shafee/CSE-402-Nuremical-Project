# N-body Solar System Simulation

Direct gravitational N-body simulation of the Solar System with relativistic corrections, benchmarking serial CPU, OpenMP, MPI, and CUDA backends.

## Physical & Astronomical Domain

**Solar System Baseline**:
The initial 10-body configuration consisting of the Sun, eight major planets, and the Moon.
_Avoid_: Solar planetary system, central star system

**Synthetic Asteroid Belt**:
Main-belt asteroid particles ($a \in [2.1, 3.3]\text{ AU}$) with synthetic Keplerian orbital elements used to scale body count $N$ up to tens of thousands.
_Avoid_: Debris field, random particles, asteroid field

**Barycentric Frame**:
The inertial reference frame centered at the center-of-mass of all simulated bodies, with zero total linear momentum.
_Avoid_: Heliocentric frame, origin frame, solar center

**Gaussian Gravitational Constant ($G$)**:
The gravitational constant expressed in Astronomical Units, solar masses, and days ($G \approx 2.9591220828559 \times 10^{-4}\text{ AU}^3 M_\odot^{-1}\text{ day}^{-2}$).
_Avoid_: SI gravity, universal G

**First-Order Post-Newtonian (1PN) Correction**:
General relativistic acceleration correction accounting for spacetime curvature, notably producing Mercury's perihelion precession of $42.98\text{ arcsec/century}$.
_Avoid_: Relativistic drift, Einsteinian force, GR effect

## Numerical & Algorithmic Domain

**Velocity Verlet Integrator**:
A second-order symplectic time-stepping scheme that updates positions and velocities with bounded energy oscillations using one force evaluation per step.
_Avoid_: Verlet algorithm, leapfrog, Euler-Verlet

**Direct Summation ($O(N^2)$)**:
Exact calculation of all pairwise gravitational force interactions without spatial tree partitioning.
_Avoid_: All-pairs, brute-force gravity, naive gravity

**Plummer Softening Length**:
A softening parameter ($\epsilon$) added to pairwise distances to eliminate mathematical singularities during close encounters (configured to $0.0$ for point masses).
_Avoid_: Collision damping, smoothing radius

**Laplace–Runge–Lenz (LRL) Vector**:
A conserved vector pointing toward the perihelion of an orbit used to monitor perihelion precession.
_Avoid_: Periapsis vector, eccentricity vector

## Architecture & Parallelization Domain

**Structure-of-Arrays (SoA)**:
A memory layout storing coordinates and masses as separate contiguous arrays (`x`, `y`, `z`, `m`) to maximize CPU cache line efficiency and SIMD vectorization.
_Avoid_: Array-of-Structures, AoS, body objects

**Newton's Third Law Variant (`newton3`)**:
An algorithmic optimization that computes pairwise interactions only for $j > p$ using $\mathbf{F}_{ij} = -\mathbf{F}_{ji}$, cutting floating-point operations by $50\%$.
_Avoid_: Pairwise symmetry, half-loop, symmetric reduction

**Dynamic Schedule Balancer (`dynamic`)**:
An OpenMP work distribution scheme assigning bodies in small blocks (e.g., chunk size 16) to balance asymmetric Intel Performance-cores (P-cores) and Efficient-cores (E-cores).
_Avoid_: Work stealing, irregular scheduling

**Shared Memory Tiling (`tiled`)**:
A GPU kernel optimization where thread blocks cooperatively stage body coordinates into on-chip `__shared__` memory, reducing global VRAM bandwidth by the block dimension factor.
_Avoid_: Block caching, GPU staging, shared tiling

**Copy-Step Negative Control (`copystep`)**:
A benchmark variant that deliberately transfers particle coordinates between host memory and GPU memory at every time step to isolate PCIe bus latency.
_Avoid_: Host round-trip, bus test, latency baseline

## Benchmark & Data Contract Domain

**Contract v1**:
The strict standardized specification governing command-line interface arguments, Structure-of-Arrays I/O, initial condition formats, and output JSON schemas.
_Avoid_: Spec v1, data schema, benchmark agreement

**Anchor Baseline**:
The single-thread scalar execution time used as the denominator ($T_{\text{serial}}$) for calculating speedup and parallel efficiency across parallel backends.
_Avoid_: Reference time, base run, baseline clock

**Correctness Gate (M1)**:
The verification check requiring relative position and velocity errors between any parallel backend and the golden reference trajectory to stay below $10^{-10}$.
_Avoid_: Sanity check, tolerance test, unit validation
