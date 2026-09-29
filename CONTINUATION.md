# CSE 402: Direct N-Body Gravitational HPC Simulation — Oral Defense Continuation Context

> **Project Evaluation:** Week 14 Final Evaluation / Oral Defense  
> **Course:** CSE 402 (High-Performance Computing / Numerical Methods)  
> **Group:** C_G8  
> **Active Git Branch:** `feature/unified-backends`  
> **Defense Server URL:** `http://localhost:8000/bench/defense_dashboard.html`  
> **Last Updated:** 2026-09-29  

---

## 1. Quick-Start & Operational Runbook

### A. Launching the Defense Platform
The defense platform is designed to run seamlessly on any standard Windows machine without requiring external dependencies, compilers, or build chains:

```bash
# From repository root
python bench/defense_server.py
```

- **Open in Browser:** Navigate to `http://localhost:8000/bench/defense_dashboard.html` (Chrome or Edge recommended).
- **Server Health Check:** The top status bar will illuminate green:
  `C++ OpenMP Engine: 12 Hardware Threads Active` (or whatever max threads the host CPU provides).
- **Zero-Dependency Portability:** `backends/openmp/build/nbody_omp_engine.dll` is statically linked (`-static -fopenmp -O3`) and tracked in git. It runs on any x86_64 Windows installation directly via Python's standard `ctypes`.

### B. Recompilation Recipe (If Modifying C++ Code)
If you edit `backends/openmp/nbody_omp_engine.cpp`:

> **IMPORTANT:** On Windows, terminate the running `defense_server.py` before compiling, otherwise the file lock on `nbody_omp_engine.dll` will produce a linker error (`permission denied`).

```bash
# 1. Ensure build directory exists
mkdir -p backends/openmp/build

# 2. Windows (MinGW-w64 / GCC with OpenMP):
g++ -O3 -fopenmp -shared -static -std=c++17 \
    backends/openmp/nbody_omp_engine.cpp \
    -o backends/openmp/build/nbody_omp_engine.dll

# 3. Linux / WSL2:
g++ -O3 -fopenmp -shared -fPIC -std=c++17 \
    backends/openmp/nbody_omp_engine.cpp \
    -o backends/openmp/build/libnbody_omp_engine.so

# 4. Relaunch server
python bench/defense_server.py
```

---

## 2. Platform Architecture & The Three Defense Tabs

```
                           [ defense_dashboard.html ]
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
   [ 1. Live Simulation ]     [ 2. Defense Slides ]    [ 3. Benchmark Explorer ]
    Dual-Pane Comparative      4 Oral Defense Slides    6-Chart Carousel
    Lockstep (0.000 ms delta)  KaTeX Mathematical Q&A   316 Empirical Runs
    Euler vs C++ OpenMP        Symplectic & Amdahl      Ryzen 5600G 6-Point Curves
            │
    /api/sim/batch (HTTP JSON)
            │
   [ defense_server.py ] ──(ctypes)──> [ nbody_omp_engine.dll ]
                                        • newton3 (O(N^2/2) halving)
                                        • simd (AVX2 branchless)
                                        • dynamic (guided load steal)
                                        • static (canonical block)
```

### Tab 1: Live Comparative Simulation
- **Dual-Pane Comparative Architecture:**
  - **Left Pane (Baseline Control):** Non-symplectic integration — Forward Euler ($\mathcal{O}(\Delta t)$ truncation error) or 4th-Order Classical Runge-Kutta (RK4, $\mathcal{O}(\Delta t^4)$ single-step truncation error but non-area-preserving).
  - **Right Pane (Team HPC Engine):** Statically compiled C++ OpenMP Velocity Verlet (`newton3`, `simd`, `dynamic`, `static`), running at hardware maximum threads (12 threads on Ryzen 5 5600G).
- **Physical Lockstep Guarantee:**
  - Synchronized via a single global master clock (`globalPhysicalTime`).
  - Both panes integrate with identical step sizes (`globalDt`), guaranteeing a physical time difference of exactly **$0.000\text{ ms}$**.
- **Continuous Forward Streaming:**
  - Client maintains streaming cursors (`streamPos`, `streamVel`) at the forward edge of the simulation.
  - Every batch request passes leading-edge positions and velocities to the DLL, eliminating position resetting or frozen velocity loops.
  - Smooth 60 FPS interpolation runs from the buffer; if network buffers deplete, Pane B seamlessly runs local Velocity Verlet fallback to preserve phase space continuity.
- **Physical Scenarios Available:**
  1. *Figure-8 3-Body* (Chenciner & Montgomery choreography — ideal for phase space stability proof)
  2. *Sun-Earth-Moon* (Hierarchical 3-body system)
  3. *Lagrange Points* (Restricted 3-body Trojan asteroids)
  4. *Solar System* (8-planet orbital scale)
  5. *Chaotic 5-Body Dance* (Demonstrates extreme sensitivity to non-symplectic drift)

### Tab 2: Defense Slides & Oral Technical Q&A
- **Slide 1 — Symplectic Geometry & Phase Space Conservation:**
  - KaTeX formal derivations of Hamiltonian vector fields, Liouville’s Theorem ($d\omega = 0$), and symplectic area preservation.
  - Mathematical proof of why Forward Euler experiences secular energy drift ($\frac{dE}{dt} \neq 0$), why RK4 eventually spirals, and why Velocity Verlet preserves a shadow Hamiltonian $\tilde{H} = H + \mathcal{O}(\Delta t^2)$ with zero long-term secular drift.
- **Slide 2 — Shared-Memory Parallelization (OpenMP):**
  - Race condition mitigation: Pairwise Newton's 3rd Law ($a_{ij} = -a_{ji}$) creates write collisions if threads update target particle $j$ simultaneously.
  - Solution: Thread-private accumulation matrices (`acc_priv[T][N][3]`) with $\mathcal{O}(N)$ parallel reduction.
  - AVX2 Vectorization (`simd`): Branchless inner loops processing 4 double-precision floats concurrently with Fused Multiply-Add (FMA3).
- **Slide 3 — Distributed HPC & Network Communication (MPI):**
  - Ring Allgather vs Master-Worker topology analysis.
  - Communication-to-Computation ratio $\mathcal{O}(1/N)$ and network latency bounds under Amdahl’s and Gustafson’s Laws.
- **Slide 4 — GPU Acceleration & Heterogeneous Scaling (CUDA):**
  - Shared memory tile caching ($B \times B$ thread blocks), avoidance of uncoalesced global memory transactions, and warp divergence minimization.
- **Interactive Defense Q&A:**
  - 8 pre-crafted, rigorous academic answers addressing likely supervisor queries (numerical stability, cache thrashing, MPI collective overheads, float precision vs performance).

### Tab 3: Benchmark Explorer
- **Interactive 6-Chart Carousel (Chart.js):**
  1. *Macro-Scaling Across Architectures:* Serial, OpenMP (Ryzen 5600G & Core i5-11400H), MPI Ring, MPI Master-Worker, CUDA RTX 5090.
  2. *OpenMP Variants Duel (Ryzen 5600G):* Full 6-point resolution curves ($N \in [100, 500, 1000, 2000, 5000, 10000]$) comparing `newton3`, `simd`, `dynamic`, `static`, and serial baseline.
  3. *MPI Distributed Topologies:* Ring Allgather vs Master-Worker scaling up to 16 ranks.
  4. *CUDA GPU Tiling Speedup:* RTX 5090 vs CPU baselines up to $N = 100,000$.
  5. *Cross-Microarchitecture Comparison:* Zen 3 vs Raptor Lake vs Apple M4.
  6. *Energy Drift & Long-term Symplectic Stability:* Forward Euler vs RK4 vs Velocity Verlet.
- **Hover & Interaction Polish:**
  - Canvas hover opacity dimming has been disabled; cursor hover triggers crisp, non-sticky inspection tooltips (`mode: 'index', intersect: false`).
  - Hardware family isolation is controlled cleanly via top preset filter buttons and interactive legend toggles.
- **Master Benchmark Matrix:**
  - Ingests empirical measurements from `bench/results/consolidated_benchmarks.json` (316 validated runs).

---

## 3. Physical Benchmark Data Reference (Ryzen 5 5600G, 12 Threads)

Median step latency ($\text{ms} / \text{step}$) across problem size $N$:

| Problem Size ($N$) | Serial Baseline | OpenMP static | OpenMP dynamic | OpenMP simd (AVX2) | OpenMP newton3 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$N = 100$** | $0.036\text{ ms}$ | $0.138\text{ ms}$ | $0.145\text{ ms}$ | $0.146\text{ ms}$ | $0.208\text{ ms}$ |
| **$N = 500$** | $0.850\text{ ms}$ | $0.328\text{ ms}$ | $0.300\text{ ms}$ | $0.232\text{ ms}$ | $0.344\text{ ms}$ |
| **$N = 1,000$** | $3.491\text{ ms}$ | $1.737\text{ ms}$ | $0.756\text{ ms}$ | $0.483\text{ ms}$ | $0.599\text{ ms}$ |
| **$N = 2,000$** | $13.899\text{ ms}$ | $3.178\text{ ms}$ | $2.572\text{ ms}$ | $1.622\text{ ms}$ | $2.163\text{ ms}$ |
| **$N = 5,000$** | $85.669\text{ ms}$ | $18.406\text{ ms}$ | $15.212\text{ ms}$ | $8.585\text{ ms}$ | $11.355\text{ ms}$ |
| **$N = 10,000$** | $338.946\text{ ms}$ | $66.784\text{ ms}$ | $64.504\text{ ms}$ | **$33.141\text{ ms}$** | $44.168\text{ ms}$ |

### High-Yield Oral Defense Defense Talking Points:
1. **The SIMD vs Newton-3 Inversion:**
   - *Supervisor Question:* "Why does `simd` run faster than `newton3` at $N=10,000$ ($33.14\text{ ms}$ vs $44.17\text{ ms}$), even though Newton's 3rd Law evaluates half the pairwise forces?"
   - *Answer:* "Newton's 3rd Law introduces a dependency because particle $j$ receives force from particle $i$. In multithreaded OpenMP, preventing write race conditions requires each thread to maintain private accumulator buffers, followed by an $\mathcal{O}(T \cdot N)$ reduction step that incurs cache misses and memory traffic. In contrast, `simd` calculates all $N^2$ interactions independently; its inner loop is completely branchless and maps directly into AMD Zen 3 AVX2 256-bit FMA vector units (4 double-precision operations per cycle per pipeline). The pure vector throughput overcomes the $2\times$ arithmetic overhead."
2. **Small $N$ Parallel Overhead:**
   - At $N=100$, Serial ($0.036\text{ ms}$) outperforms OpenMP ($0.138\text{ ms}$ – $0.208\text{ ms}$) due to thread fork/join synchronization and barrier latency dominating over the trivial computation ($100^2 = 10,000$ iterations). The parallel crossover point occurs between $N=200$ and $N=300$.

---

## 4. Key Defense Presentation Constraints & Design Policies

1. **Academic Integrity & Evaluated Physics:**
   - Only evaluated, genuine implementations are present in the UI. No fictional physics (no unverified 1PN General Relativistic precession or Yoshida-4 integrators).
   - Comparative baselines are authentic: Forward Euler (1st order explicit), Classical RK4 (4th order explicit), and Velocity Verlet (2nd order symplectic).
2. **Visual Standards:**
   - Strictly emoji-free. All iconography uses SVG outlines with academic typography (Inter, JetBrains Mono, KaTeX math rendering).
   - Unified Observatory Dark palette (`#06090e`, `#0d1522`, `#162238`, `#38bdf8`, `#10b981`, `#fb7185`).
3. **Synchronization & State Integrity:**
   - Single global $\Delta t$ control slider. Individual pane desynchronization controls have been excised.
   - Dual-pane time delta is hard-locked at $0.000\text{ ms}$.

---

## 5. Verification & Test Suite

All verification scripts are located in `scratch/`:
- `scratch/verify_sync_v9.js`: Connects to Chrome over DevTools Protocol, verifies live orbital animation, confirms dual-pane lockstep time delta ($0.000\text{ ms}$), and measures energy drift on both panes.
- `scratch/check_console.js`: Validates zero runtime JavaScript warnings or errors in the browser console.
- `scratch/test_api_batch.py`: Performs HTTP POST integration tests on `/api/sim/batch` for all 4 OpenMP variants (`newton3`, `simd`, `dynamic`, `static`).

---

## 6. Git Status & Next Steps

- **Branch:** `feature/unified-backends`
- **Key Modified Files:**
  - `bench/defense_dashboard.html`: Complete live simulation, defense slides, and benchmark explorer.
  - `bench/defense_server.py`: Python daemon with ctypes bridge.
  - `backends/openmp/nbody_omp_engine.cpp`: Native C++ multi-threaded force engine.
  - `backends/openmp/build/nbody_omp_engine.dll`: Portable Windows binary.
  - `.gitignore`: Configured to preserve portable DLL while ignoring scratch build logs.
  - `CONTINUATION.md`: This comprehensive defense briefing.

**Ready for Defense Rehearsal:**
Run `python bench/defense_server.py`, open `http://localhost:8000/bench/defense_dashboard.html`, and proceed through the live demonstration and slide presentation.
