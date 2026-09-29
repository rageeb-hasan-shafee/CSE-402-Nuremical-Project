# CSE 402: Direct N-Body Gravitational HPC Simulation — Oral Defense Continuation Context

> **Project Evaluation:** Week 14 Final Evaluation / Oral Defense  
> **Group:** C_G8  
> **Git Branch:** `feature/unified-backends`  
> **Defense Server URL:** `http://localhost:8000/bench/defense_dashboard.html`  
> **Last Updated:** 2026-09-29  

---

## 1. Executive Summary & Current State

The **BUET CSE 402 Oral Defense Demonstration Platform** has been consolidated and aligned under Contract v1. The platform integrates real-time comparative gravitational orbital mechanics with empirical HPC benchmark analysis across heterogeneous architectures (AMD Zen 3, Intel Raptor Lake, Apple M4, and NVIDIA RTX 5090 Blackwell).

### Core Components
1. **Interactive Oral Defense Hub** (`bench/defense_dashboard.html`):
   - **Tab 1: Live Simulation** — Dual-pane physical simulator comparing baseline non-symplectic controls (Forward Euler, RK4) against the team's genuine C++ OpenMP Velocity Verlet engine running at 12 hardware threads.
   - **Tab 2: Defense Slides & Q&A** — 4 academic presentation slides with KaTeX mathematical formulas, physics derivations, and oral defense defense-ready responses to supervisor questions.
   - **Tab 3: Benchmark Explorer** — 6-chart interactive carousel (Chart.js) and a comprehensive 316-run performance matrix spanning Serial, OpenMP (4 variants), Distributed MPI (2 topologies), and CUDA GPU tiling.

2. **Native C++ OpenMP Live Engine** (`backends/openmp/nbody_omp_engine.cpp`):
   - Compiled with `-O3 -fopenmp -shared -static -std=c++17` into `backends/openmp/build/nbody_omp_engine.dll`.
   - Statically linked with zero runtime DLL dependencies (runnable out-of-the-box on any Windows machine).
   - Evaluates strictly the team's genuine Velocity Verlet variants:
     - `newton3`: Pairwise $\mathcal{O}(N^2/2)$ halving via Newton's 3rd Law ($a_{ij} = -a_{ji}$) with dynamic thread-private reduction buffers.
     - `simd`: AVX2 vectorization splitting the inner loop around index $p$ for branchless `#pragma omp simd` accumulation.
     - `dynamic`: Dynamic chunk scheduling (`schedule(dynamic, 16)`) for work-stealing across asymmetric loads.
     - `static`: Canonical static chunk scheduling (`schedule(static)`).

3. **Lightweight Python Defense Daemon** (`bench/defense_server.py`):
   - Serves the dashboard on `http://localhost:8000`.
   - Exposes `/api/sim/batch` which bridges JavaScript requests directly to `nbody_omp_engine.dll` using `ctypes`.
   - Returns full $(x, y, z)$ positions, $(v_x, v_y, v_z)$ velocities, and relative energy drift $\Delta E / |E_0|$ for continuous forward simulation.

---

## 2. Key Architecture Fixes & Guarantees Established

### A. Continuous Forward Momentum in OpenMP Streaming
- **The Issue Solved:** Previously, periodic HTTP batch fetches sampled the rendered canvas positions while the queue was unplayed, causing the C++ engine to loop backward in time and restart from frozen initial velocities.
- **The Fix:** `SimulationPane` maintains `streamPos` and `streamVel` at the leading edge of the simulation stream. Each batch starts exactly from `data.final_pos` and `data.final_vel` of the previous batch.
- **Velocity Synchronization:** Both positions and velocities are passed into and returned from the C++ engine at every step.
- **Symplectic Fallback:** If network delay causes the client queue to temporarily deplete, `stepNumerical(dt)` executes the team's local Velocity Verlet, never degrading into non-symplectic Forward Euler.

### B. Synchronous Physical Time Advance (0.000 ms Time Delta)
- **Single Global Physical Clock:** Driven by `globalPhysicalTime += (elapsedMs / 1000.0) * (globalSpeed * 2.5);`.
- **Lockstep Stepping:** Both Pane A and Pane B advance by `globalDt` until matching `globalPhysicalTime`, guaranteeing identical integration timestamps without pane desynchronization.

### C. Clean Interaction in Benchmark Explorer
- **Canvas Hover Decoupled from Series Dimming:** The Chart.js canvas `onHover` callback was completely removed. Moving the mouse across curves displays standard tooltip inspection (`mode: 'index', intersect: false`) without modifying line opacity or leaving the chart stuck in a dimmed state.
- **Selective Filtering:** Series isolation is cleanly driven by preset buttons (`All Paradigms`, `OpenMP Variants`, `MPI Distributed`, `CUDA GPU`, `Microarchitecture Duel`) and bottom legend toggles.

### D. Strict Academic Aesthetics
- Zero unicode emojis across tabs, sliders, badges, and slides; replaced with clean inline SVGs, KaTeX typography, and a subtle Celestial Observatory palette (`#06090e`, `#0d1522`, `#162238`, `#38bdf8`, `#10b981`, `#fb7185`).

---

## 3. Physical Benchmark Summary (AMD Ryzen 5 5600G, 12 Threads)

Measured median execution times ($\text{ms} / \text{step}$) ingested into `bench/results/consolidated_benchmarks.json`:

| Problem Size ($N$) | Serial Baseline | OpenMP static | OpenMP dynamic | OpenMP simd (AVX2) | OpenMP newton3 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$N = 100$** | $0.036\text{ ms}$ | $0.138\text{ ms}$ | $0.145\text{ ms}$ | $0.146\text{ ms}$ | $0.208\text{ ms}$ |
| **$N = 500$** | $0.850\text{ ms}$ | $0.328\text{ ms}$ | $0.300\text{ ms}$ | $0.232\text{ ms}$ | $0.344\text{ ms}$ |
| **$N = 1,000$** | $3.491\text{ ms}$ | $1.737\text{ ms}$ | $0.756\text{ ms}$ | $0.483\text{ ms}$ | $0.599\text{ ms}$ |
| **$N = 2,000$** | $13.899\text{ ms}$ | $3.178\text{ ms}$ | $2.572\text{ ms}$ | $1.622\text{ ms}$ | $2.163\text{ ms}$ |
| **$N = 5,000$** | $85.669\text{ ms}$ | $18.406\text{ ms}$ | $15.212\text{ ms}$ | $8.585\text{ ms}$ | $11.355\text{ ms}$ |
| **$N = 10,000$** | $338.946\text{ ms}$ | $66.784\text{ ms}$ | $64.504\text{ ms}$ | $33.141\text{ ms}$ | $44.168\text{ ms}$ |

*Key Insight:* On Zen 3 AVX2 registers, `simd` achieves the lowest wall-clock latency ($33.14\text{ ms}$ at $N=10,000$, a $10.2\times$ speedup over serial) due to high throughput vector pipelines. `newton3` cuts mathematical operations by 50% ($44.17\text{ ms}$), with slight overhead from thread-private reduction buffers.

---

## 4. Refinement Roadmap for Oral Defense

When returning to polish or defend:

1. **Slide Deck & Talking Points (Tab 2):**
   - Verify that KaTeX mathematical formulations in Slide 1 (Symplectic Geometry & Hamiltonian Phase Space Preservation) and Slide 3 (Amdahl's Law & Communication Scaling) match the oral defense speech script.
   - Practice the explanation for why Forward Euler exhibits catastrophic energy divergence ($\mathcal{O}(\Delta t)$ secular drift) while Velocity Verlet stays bounded ($\mathcal{O}(\Delta t^2)$ bounded energy oscillation).

2. **Server Launch Verification:**
   - To launch or restart the server on any Windows host:
     ```bash
     python bench/defense_server.py
     ```
   - Open `http://localhost:8000/bench/defense_dashboard.html` in Chrome or Edge.
   - Confirm the green pill: `C++ OpenMP Engine: 12 Threads Active`.

3. **Testing & Validation Scripts in `scratch/`:**
   - `scratch/verify_sync_v9.js` — Automated Chrome CDP audit checking exact $0.000\text{ ms}$ lockstep and energy drift.
   - `scratch/test_api_batch.py` — Tests all 4 OpenMP variants via `/api/sim/batch`.
   - `scratch/check_console.js` — Audits browser console for zero runtime exceptions.
