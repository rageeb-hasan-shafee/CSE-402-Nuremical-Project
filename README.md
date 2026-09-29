# Direct Gravitational N-Body Simulation of the Solar System with Relativistic Corrections & Heterogeneous Parallel Acceleration

> **Course:** CSE 402 - Simulation and Modeling  
> **Group:** C_G8  
> **Evaluation Track:** Oral Defense & Comprehensive Parallel Simulation Platform  
> **Base Research Paper:** Tailin Zhu, *"N-body Simulations of the Solar System with CPU-based Parallel Methods"*, University of Bristol (2020), [arXiv:2112.15079](https://arxiv.org/abs/2112.15079).

---

## Team Members & Responsibilities

| Student Name | Student ID | Specific Responsibility & Parallel Backend | Evaluated Hardware Node |
|---|---|---|---|
| **Saidul Anam Siam** | **2105156** | **Shared-Memory Parallelism (OpenMP)** & **Compiled C++ Baseline (`cpp-serial`)**: Implemented all 4 OpenMP variants (`static`, `dynamic`, `simd`, `newton3`), native C++ streaming engine DLL, and Intel P/E hybrid core analysis. | Intel® Core™ i5-1340P (4P+8E cores, 16T) |
| **Rageeb Hasan Shafee** | **2105175** | **System Architecture, Defense Dashboard & Platform Lead**: Designed Contract v1, integration hub, interactive HTML5/WebGL dual-pane oral defense visualizer, and comparative benchmark engine. | AMD Ryzen™ 5 5600G (6C/12T) |
| **Nurul Mansib Talukder** | **2105159** | **Distributed-Memory Parallelism (MPI)**: Implemented `mpi4py` distributed backends (`allgather` ring and Master-Worker dynamic task pool), cluster communication scaling, and Amdahl analysis. | Apple M4 (4P+6E cores, 10T) & Linux cluster |
| **Mustafa Muhaimin** | **2105178** | **Mathematical Modeling, Celestial Mechanics & Physical Validation**: Symplectic Velocity Verlet integrators, General Relativity 1PN corrections (Mercury precession), and asteroid belt IC generation. | Python Reference & Physical Testbed |
| **Ariful Islam Shadhin** | **2105166** | **Massive GPU Parallelism (CUDA)**: GPU acceleration with tiled `__shared__` memory ($B=256$), non-divergent warp scheduling, precision scaling (FP32 vs FP64), and PCIe transfer profiling. | NVIDIA® GeForce RTX™ 5090 (24,576 cores) & Tesla T4 |

---

## 1. Quick Start: One-Click Oral Defense Platform

The repository includes a comprehensive, browser-based **Supervisor Oral Defense & Live Demonstration Dashboard** featuring real-time C++ OpenMP streaming, dual-pane 3D/2D orbit visualization, live energy drift telemetry, and cross-paradigm benchmark exploration.

### Launch the Platform

**On Windows:**
```powershell
.\run_defense.bat
```

**On Linux / macOS:**
```bash
chmod +x run_defense.sh
./run_defense.sh
```

This starts the lightweight HTTP/WebSocket bridge (`bench/defense_server.py`) and automatically opens [`bench/defense_dashboard.html`](bench/defense_dashboard.html) in your browser at `http://localhost:8000`.

---

## 2. Repository Architecture & Directory Structure

```text
CSE-402-Nuremical-Project/
├── backends/                           # Heterogeneous simulation backends
│   ├── openmp/                         # Shared-Memory C++ OpenMP (Siam - 2105156)
│   │   ├── nbody_omp.cpp               # Master source compiling to nbody_serial & nbody_omp (4 variants)
│   │   ├── nbody_omp_engine.cpp        # Real-time C++ streaming engine (builds nbody_omp_engine.dll)
│   │   ├── CMakeLists.txt              # Standardized CMake build file (-O3, -mavx2, -fopenmp)
│   │   ├── build_mingw.bat / .ps1      # Automated Windows MinGW build scripts
│   │   ├── sweep.py                    # Automated hardware benchmark sweep runner
│   │   ├── test_m1_gate.py             # Milestone 1 automated correctness gate validator
│   │   ├── BENCHMARK_REPORT.md         # Dedicated OpenMP benchmark findings on Intel i5-1340P
│   │   └── README.md                   # Standalone OpenMP documentation
│   ├── mpi/                            # Distributed-Memory MPI (Mansib - 2105159)
│   │   ├── nbody_mpi.py                # MPI simulation implementing allgather and master-worker
│   │   ├── sweep_local.py              # Automated MPI benchmark driver across ranks
│   │   ├── verify_mpi.py               # MPI correctness gate validation
│   │   └── README.md                   # Standalone MPI documentation
│   ├── cuda/                           # Massive-scale GPU Acceleration (Shadhin - 2105166)
│   │   ├── nbody_cuda.cu               # CUDA kernel (naive, tiled shared-memory, copystep)
│   │   ├── sweep_cuda.py               # CUDA benchmark driver across grid dimensions & precisions
│   │   ├── colab_run.ipynb             # Google Colab cloud execution notebook
│   │   └── README.md                   # Standalone CUDA documentation
│   ├── serial/                         # Dedicated C++ Scalar Anchor
│   │   └── nbody_serial.cpp            # Clean single-threaded C++ baseline
│   └── python_ref/                     # Python Ground-Truth Reference
│       └── nbody_ref.py                # Pure NumPy vectorized baseline producing golden states
│
├── bench/                              # Benchmarking framework & evaluation results
│   ├── results/                        # 1,000+ verified Contract v1 benchmark JSONs
│   │   ├── cpp-openmp/                 # OpenMP runs (siam-i5-1340p, fahad-ryzen-5600g)
│   │   ├── cpp-serial/                 # Serial C++ runs (siam-i5-1340p, fahad-ryzen-5600g)
│   │   ├── py-mpi/                     # MPI runs across ranks (mansib-m4)
│   │   └── py-numpy/                   # NumPy reference runs (fahad-ryzen-5600g)
│   ├── plots/                          # High-resolution generated publication figures (Fig 1-6)
│   ├── defense_dashboard.html          # Interactive Oral Defense platform
│   ├── defense_server.py               # Live telemetry & simulation streaming bridge
│   ├── generate_plots_and_report.py    # Automated report & publication chart generator
│   ├── MASTER_BENCHMARK_REPORT.md      # Comprehensive cross-architecture comparative report
│   ├── validate.py                     # Milestone 1 floating-point correctness gate validator
│   └── check_result.py                 # Schema v1 JSON format validator
│
├── common/                             # Shared contracts & cross-language I/O utilities
│   ├── CONTRACT.md                     # Official Contract v1 specification (CLI, SoA, schema)
│   ├── cpp/nbody_io.hpp                # Zero-allocation Structure-of-Arrays (SoA) C++ I/O header
│   ├── python/nbio.py                  # Contract v1 state reader/writer for Python
│   └── python/export_ic.py             # Synthetic asteroid belt initial condition generator
│
├── data/                               # Initial conditions & analytical ground-truth trajectories
│   ├── ic/                             # Initial condition CSVs (N = 100, 500, 1000, 2000, 5000, 10000, 20000, 50000)
│   └── ref/                            # Golden reference states for M1 correctness gating
│
├── docs/                               # Architecture decision records & guides
│   ├── adr/                            # Architecture Decision Records (ADR 0001, ADR 0002)
│   ├── IMPLEMENTATION_GUIDE_*.md       # Individual technical roadmaps for each student role
│   └── visual_design_research.md       # Visual design system specifications
│
├── nbody_sim/                          # Physical modeling, orbital mechanics & GR testbed
│   ├── simulation.py                   # Core symplectic integration with chunked force evaluation
│   ├── orbital_elements.py             # JPL Horizons J2000 Keplerian orbital elements converter
│   ├── onepn.py                        # General Relativity First-Order Post-Newtonian (1PN) engine
│   ├── run_mercury_precession_test.py  # Einstein's 43"/century Mercury precession verification
│   └── run_accuracy_test.py            # Long-term symplectic energy conservation validation
│
├── run_defense.bat                     # Windows one-click defense launcher
├── run_defense.sh                      # Linux/macOS one-click defense launcher
└── README.md                           # Master group documentation
```

---

## 3. Physical & Mathematical Modeling

### 3.1. Symplectic Velocity Verlet Integration
Planetary orbits require numerical integrators that preserve phase-space volume and exhibit zero secular energy drift over astronomical timescales. We implement the second-order **Velocity Verlet** scheme:

$$\mathbf{r}_i(t + \Delta t) = \mathbf{r}_i(t) + \mathbf{v}_i(t)\Delta t + \frac{1}{2}\mathbf{a}_i(t)\Delta t^2$$

$$\mathbf{a}_i(t + \Delta t) = \sum_{j \neq i} \frac{G m_j (\mathbf{r}_j - \mathbf{r}_i)}{\left(|\mathbf{r}_j - \mathbf{r}_i|^2 + \epsilon^2\right)^{3/2}} + \mathbf{a}_i^{\text{1PN}}$$

$$\mathbf{v}_i(t + \Delta t) = \mathbf{v}_i(t) + \frac{1}{2}\left[\mathbf{a}_i(t) + \mathbf{a}_i(t + \Delta t)\right]\Delta t$$

where $G = 2.9591220828559 \times 10^{-4}\text{ AU}^3 M_\odot^{-1}\text{ day}^{-2}$ is the Gaussian gravitational constant, and $\epsilon$ is the Plummer softening parameter ($\epsilon = 0.0$ for celestial point masses).

### 3.2. General Relativity: First-Order Post-Newtonian (1PN) Corrections
To capture relativistic perihelion precession without introducing full numerical relativity, we implement the 1PN acceleration correction in [`nbody_sim/onepn.py`](nbody_sim/onepn.py):

$$\mathbf{a}_{\text{1PN}} = \frac{G M_\odot}{c^2 r^3} \left[ \left( \frac{4 G M_\odot}{r} - v^2 \right) \mathbf{r} + 4 (\mathbf{r} \cdot \mathbf{v}) \mathbf{v} \right]$$

- **Physical Verification:** Running `python nbody_sim/run_mercury_precession_test.py` validates Mercury's anomalous perihelion advance of **$42.98\text{ arcsec/century}$**, matching Einstein's celebrated 1915 result.
- **Symplectic Energy Conservation:** Running `python nbody_sim/run_accuracy_test.py` proves total energy error $|\Delta E / E_0| < 10^{-5}$ over thousands of orbits.

---

## 4. Backend Implementation & Execution Guides

All backends strictly adhere to **Contract v1** ([`common/CONTRACT.md`](common/CONTRACT.md)) and support uniform CLI parameters:
`--input <file> --steps <int> --dt <float> --workers <int> --variant <str> --output-json <file> --final-state <file>`

### 4.1. Shared-Memory C++ OpenMP & Serial Baseline (`cpp-openmp`, `cpp-serial`)
*Lead: Saidul Anam Siam (2105156)*

#### A. Compilation
From `backends/openmp/`:
```powershell
# Using the Windows MinGW build script:
.\build_mingw.bat

# Or using standard CMake:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```
This produces:
- `nbody_serial.exe`: The single-threaded compiled C++ baseline anchor.
- `nbody_omp.exe`: The multi-threaded OpenMP parallel engine.
- `nbody_omp_engine.dll`: The shared library driving real-time browser simulation streaming.

#### B. The 4 OpenMP Algorithmic Variants
1. **`static`**: Equal block scheduling ($N/P$ bodies per thread). Maximum cache locality on symmetric cores.
2. **`dynamic`**: Work queue with chunk size 16. Crucial for asymmetric architectures (e.g. Intel P-cores vs E-cores), preventing fast threads from starving at the barrier.
3. **`simd`**: Branch-free inner loop with OpenMP SIMD pragmas, leveraging 256-bit AVX2 vector registers.
4. **`newton3`**: Exploits Newton's 3rd Law ($\mathbf{F}_{ji} = -\mathbf{F}_{ij}$), computing only $j > p$ and accumulating forces into thread-private buffers. Cuts arithmetic operations by exactly $50\%$.

#### C. Running OpenMP Simulations
```powershell
# Run serial baseline:
.\build\nbody_serial.exe --input ../../data/ic/ic_N1000_s42.csv --steps 100 --dt 0.1

# Run OpenMP dynamic with 4 threads:
.\build\nbody_omp.exe --input ../../data/ic/ic_N1000_s42.csv --steps 100 --dt 0.1 --workers 4 --variant dynamic

# Run OpenMP newton3 with 16 threads:
.\build\nbody_omp.exe --input ../../data/ic/ic_N5000_s42.csv --steps 100 --dt 0.1 --workers 16 --variant newton3
```

#### D. Running Automated OpenMP Sweeps
```powershell
python sweep.py --quick     # Fast sanity sweep (N = 100, 500, 1000)
python sweep.py             # Full sweep across all N and thread counts
```

---

### 4.2. Distributed-Memory MPI Backend (`py-mpi`)
*Lead: Nurul Mansib Talukder (2105159)*

Implements distributed multi-process parallelization using `mpi4py`:
- **`allgather` Ring Topology:** Each process computes forces for an $N/P$ slice of bodies, then exchanges updated state vectors via non-blocking collective communication (`MPI_Allgather`).
- **`master-worker` Dynamic Queue:** Master node dynamically doles out work chunks to worker ranks, demonstrating load balancing over high-latency networks.

#### Running MPI Simulations
```bash
# Run with 4 MPI ranks:
mpiexec -n 4 python backends/mpi/nbody_mpi.py --input data/ic/ic_N1000_s42.csv --steps 100 --dt 0.1 --variant allgather

# Run automated rank scaling sweep:
python backends/mpi/sweep_local.py
```

---

### 4.3. GPU-Accelerated CUDA Backend (`cuda`)
*Lead: Ariful Islam Shadhin (2105166)*

Implemented in [`backends/cuda/nbody_cuda.cu`](backends/cuda/nbody_cuda.cu) for massive $N$ scalability ($N \ge 10,000$):
- **Tiled Shared-Memory Kernel (`tiled`):** Loads body position chunks into `__shared__` memory ($B = 256$ threads per block), reducing global VRAM memory transactions by $256\times$.
- **Zero PCIe Overhead (`copystep`):** Keeps simulation state entirely in device memory across all time steps, transferring coordinates back only for visualization frames.
- **Precision Modes:** Evaluated in both FP64 (scientific double precision) and FP32 (fast single precision).

#### Running CUDA Simulations
```bash
# Compile on Linux / Google Colab with NVCC:
nvcc -O3 -arch=sm_80 backends/cuda/nbody_cuda.cu -o backends/cuda/nbody_cuda

# Execute on GPU:
./backends/cuda/nbody_cuda --input data/ic/ic_N10000_s42.csv --steps 100 --dt 0.1 --variant tiled-f32
```
*Note: A ready-to-run Google Colab notebook is provided in [`backends/cuda/colab_run.ipynb`](backends/cuda/colab_run.ipynb).*

---

### 4.4. Python Reference Baseline (`python_ref`)
*Lead: Mustafa Muhaimin (2105178)*

A vectorised NumPy implementation ([`backends/python_ref/nbody_ref.py`](backends/python_ref/nbody_ref.py)) serving as the golden mathematical ground-truth:
```bash
python backends/python_ref/nbody_ref.py --input data/ic/ic_N1000_s42.csv --steps 10 --dt 0.1 --final-state data/ref/final_N1000_steps10.csv
```

---

## 5. Correctness & Verification Gates

The repository enforces strict verification gates to guarantee physical fidelity and numerical correctness across all parallel implementations.

### 5.1. The M1 Numerical Correctness Gate
Compares simulated endpoint state vectors $(\mathbf{x}, \mathbf{y}, \mathbf{z}, \mathbf{v}_x, \mathbf{v}_y, \mathbf{v}_z)$ against the golden reference state after 10 full steps:

$$\text{Relative Error} = \max_{i} \frac{\|\mathbf{r}_i^{\text{test}} - \mathbf{r}_i^{\text{ref}}\|}{\|\mathbf{r}_i^{\text{ref}}\|} < 10^{-10}$$

To run the automated validation suite:
```powershell
python backends/openmp/test_m1_gate.py
```
*All 4 OpenMP variants (`static`, `dynamic`, `simd`, `newton3`) pass with machine-precision accuracy ($< 2 \times 10^{-16}$).*

### 5.2. Schema Validation Gate
Every benchmark output JSON is verified for schema integrity:
```powershell
python bench/check_result.py bench/results/cpp-openmp/siam-i5-1340p/cpp-openmp_static_N1000_P4_r1.json
```

---

## 6. Master Benchmark Results & Cross-Paradigm Analysis

Over **1,000 independent benchmark executions** were conducted across diverse CPU and GPU microarchitectures.

### 6.1. Cross-Architecture Performance Table (ms / step)

| Problem Size ($N$) | AMD Ryzen 5600G Serial (C++) | Intel i5-1340P Serial (C++) | OpenMP 12T (Ryzen 5600G) | OpenMP 16T (`newton3` i5) | MPI 10P (Apple M4) | NVIDIA RTX 5090 (Tiled FP64) | NVIDIA RTX 5090 (Tiled FP32) | Maximum Parallel Speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$N = 100$** | **0.036 ms** | 0.147 ms | 0.138 ms | 1.195 ms | 0.091 ms | 0.094 ms | 0.012 ms | **2.9×** |
| **$N = 500$** | 0.850 ms | 3.818 ms | 0.328 ms | 0.950 ms | 1.308 ms | 0.435 ms | 0.034 ms | **24.7×** |
| **$N = 1,000$** | 3.491 ms | 14.920 ms | 1.737 ms | **2.610 ms** | 4.328 ms | 0.865 ms | 0.063 ms | **55.3×** |
| **$N = 2,000$** | 13.899 ms | 59.850 ms | 3.178 ms | **9.850 ms** | 17.466 ms | 1.725 ms | 0.118 ms | **117.8×** |
| **$N = 5,000$** | 85.669 ms | 374.500 ms | 18.406 ms | **24.480 ms** | 127.661 ms | 4.306 ms | 0.281 ms | **304.4×** |
| **$N = 10,000$** | 338.946 ms | 1489.600 ms | 66.783 ms | **68.200 ms** | 483.054 ms | 34.333 ms | **0.556 ms** | **609.4×** |
| **$N = 50,000$** | — | — | — | — | — | — | **3.090 ms** | **> 2,000×** |

---

### 6.2. Key Academic Findings & Architectural Phenomena

#### 1. Small-$N$ Thread Creation Overhead (Replicating Zhu 2020 Fig. 4)
At $N = 100$ ($10^4$ pairwise operations), single-threaded execution consistently beats multi-threading:
- Single-thread runtime: **$0.168\text{ ms/step}$** on Intel i5-1340P.
- 16-thread OpenMP runtime: **$0.704\text{ ms/step}$** ($4.2\times$ slower).
- **Explanation:** Thread dispatching, fork/join barriers, and L1 cache invalidation take longer than computing $10^4$ floating-point instructions. Parallel execution should strictly be engaged when $N \ge 500$.

#### 2. Asymmetric Hybrid Core Load Balancing (Intel P-cores vs. E-cores)
- The Intel Core i5-1340P has 4 Performance-cores and 8 Efficient-cores.
- Under `static` scheduling at $N=5,000$, threads assigned to P-cores finish early and idle at the barrier waiting for E-cores ($44.20\text{ ms/step}$).
- Under `dynamic` scheduling, fast P-cores continuously retrieve 16-body chunks from the queue, naturally absorbing the bulk of the workload and achieving **$30.55\text{ ms/step}$ ($1.45\times$ speedup over static)**.

#### 3. Newton's 3rd Law Pair Reduction (`newton3`)
- Evaluating only the upper triangle ($j > p$) cuts pairwise force calculations from $N(N-1)$ to $\frac{N(N-1)}{2}$.
- On Intel Core i5-1340P with 16 threads, `newton3` delivers **$24.48\text{ ms/step}$ at $N=5,000$**, processing **over $1.02 \times 10^9$ pairs/second**.

#### 4. GPU Scalability & Precision Disparity (NVIDIA RTX 5090)
- **Tiled Shared-Memory Gain:** Loading coordinates into `__shared__` memory cuts global VRAM latency, boosting performance by up to $18\times$ over naive global memory access.
- **Precision Divergence:** Consumer GPUs feature fewer FP64 double-precision ALUs compared to FP32 single-precision ALUs. Single-precision (`tiled-f32`) executes **$15\times$ to $28\times$ faster** than double-precision (`tiled-f64`), achieving **$0.556\text{ ms/step}$ at $N=10,000$**.

---

### 6.3. Crossover Hierarchy: Which Backend Wins Where?

```text
  N < 100            100 <= N < 5,000          5,000 <= N < 10,000          N >= 10,000
┌──────────────┐    ┌────────────────────┐    ┌──────────────────────┐    ┌────────────────┐
│  cpp-serial  │───>│  cpp-openmp (C++)  │───>│  openmp (newton3)    │───>│  CUDA GPU      │
│  (No barrier │    │  (Shared memory    │    │  (50% pair reduction │    │  (Massive warp │
│   latency)   │    │   zero overhead)   │    │   dominates memory)  │    │   parallelism) │
└──────────────┘    └────────────────────┘    └──────────────────────┘    └────────────────┘
```

---

## 7. Generated Publication Visualizations

Publication-ready figures are generated under [`bench/plots/`](bench/plots/):

| Figure | Description | Key Insight |
|---|---|---|
| **Fig 1: Log-Log Time vs $N$** | [`bench/plots/fig1_time_vs_n_loglog.png`](bench/plots/fig1_time_vs_n_loglog.png) | Demonstrates strict asymptotic $O(N^2)$ slope across all backends. |
| **Fig 2: Speedup vs $N$** | [`bench/plots/fig2_speedup_vs_n.png`](bench/plots/fig2_speedup_vs_n.png) | Highlights backend crossover from serial to OpenMP, MPI, and CUDA. |
| **Fig 3: OpenMP Thread Scaling** | [`bench/plots/fig3_openmp_thread_scaling.png`](bench/plots/fig3_openmp_thread_scaling.png) | Strong scaling curves showing P-core saturation and SMT knee. |
| **Fig 4: OpenMP Variants** | [`bench/plots/fig4_openmp_variants.png`](bench/plots/fig4_openmp_variants.png) | Direct comparison of `static`, `dynamic`, `simd`, and `newton3`. |
| **Fig 5: MPI Communication Scaling** | [`bench/plots/fig5_mpi_scaling_and_overhead.png`](bench/plots/fig5_mpi_scaling_and_overhead.png) | Communication-to-computation ratio and network bottleneck scaling. |
| **Fig 6: CUDA Tiling & Precision** | [`bench/plots/fig6_cuda_tiling_and_precision.png`](bench/plots/fig6_cuda_tiling_and_precision.png) | Shared memory tiling speedup and FP32 vs FP64 throughput ratio. |

To regenerate all figures and markdown reports from raw JSON data:
```powershell
python bench/generate_plots_and_report.py
```

---

## 8. Architecture Decision Records (ADRs)

Key architectural decisions are documented under [`docs/adr/`](docs/adr/):
- **[ADR 0001: Unified Architecture Contract and Benchmark Ingestion](docs/adr/0001-unification-and-benchmarks.md)**: Standardizing Contract v1 across diverse team microarchitectures.
- **[ADR 0002: Dual Anchoring & Oral Defense Dashboard Architecture](docs/adr/0002-dual-anchoring-and-defense-dashboard.md)**: Rationale for native C++ streaming and dual anchoring against `cpp-serial` and `py-numpy`.
