# GPU-Accelerated N-Body Simulation of the Solar System

> **Course:** CSE 402 - Simulation and Modeling  
> **Group:** C_G8  
> **Team Members:**  
> - Saidul Anam Siam (2105156) — *OpenMP & Compiled Serial Baseline (`cpp-serial`)*  
> - Rageeb Hasan Shafee (2105175) — *Integration, Benchmarks & Demo*  
> - Nurul Mansib Talukder (2105159) — *MPI Distributed Parallelism*  
> - Mustafa Muhaimin (2105178) — *Modeling & Testing*  
> - Ariful Islam Shadhin (2105166) — *CUDA GPU Parallelism*  
>  
> **Base Paper:** Tailin Zhu, *"N-body Simulations of the Solar System with CPU-based Parallel Methods"*, University of Bristol (2020), arXiv:2112.15079.

---

## 1. Quick Start: How to Run (Step-by-Step)

Open your terminal (PowerShell or Command Prompt) in `CSE-402-Nuremical-Project`:

### Step 1: Build the C++ Executables
From the `CSE-402-Nuremical-Project` root directory:

**In PowerShell:**
```powershell
cd backends\openmp
.\build_mingw.bat
cd ..\..
```

**In Command Prompt (cmd.exe):**
```cmd
cd backends\openmp
build_mingw.bat
cd ..\..
```
This compiles two executables into `backends/openmp/build/`:
- `nbody_serial.exe` (The single-threaded compiled CPU baseline)
- `nbody_omp.exe` (The multi-threaded OpenMP parallel engine)

---

### Step 2: Run a Single Simulation

#### A. Run the Compiled Serial Baseline (`cpp-serial`)
Simulate 1,000 bodies for 100 steps:
```powershell
.\backends\openmp\build\nbody_serial.exe --input data/ic/ic_N1000_s42.csv --steps 100 --dt 0.1
```
*Output:*
```text
cpp-serial[static] N=1000 P=1 steps=100 loop=1.4920s (14.920 ms/step)
```

#### B. Run the OpenMP Parallel Engine (`cpp-openmp`)
Simulate 1,000 bodies with **4 threads** using the **dynamic** schedule:
```powershell
.\backends\openmp\build\nbody_omp.exe --input data/ic/ic_N1000_s42.csv --steps 100 --dt 0.1 --workers 4 --variant dynamic
```
*Output:*
```text
cpp-openmp[dynamic] N=1000 P=4 steps=100 loop=0.4120s (4.120 ms/step)  --> (3.6x faster!)
```

Simulate with the **`newton3`** variant (cuts math calculations in half):
```powershell
.\backends\openmp\build\nbody_omp.exe --input data/ic/ic_N1000_s42.csv --steps 100 --dt 0.1 --workers 4 --variant newton3
```
*Output:*
```text
cpp-openmp[newton3] N=1000 P=4 steps=100 loop=0.2610s (2.610 ms/step)  --> (5.7x faster!)
```

#### C. Run the Python Reference Baseline
```powershell
python backends/python_ref/nbody_ref.py --input data/ic/ic_N1000_s42.csv --steps 10
```

---

### Step 3: Run the Verification Correctness Gate
To verify that all C++ variants match the analytical ground-truth to within floating-point tolerance ($< 10^{-10}$ relative error):
```powershell
python backends/openmp/test_m1_gate.py
```
*Expected output: `ALL M1 CORRECTNESS GATE CHECKS PASSED PERFECTLY!`*

---

### Step 4: Run the Automated Benchmark Sweeps
To run automated benchmarks across body counts ($N$), thread counts ($P$), and algorithmic variants:
```powershell
# Quick sanity sweep (N = 100, 500, 1000)
python backends/openmp/sweep.py --quick

# Full production sweep (N = 100 to 10,000, threads 1 to 24)
python backends/openmp/sweep.py
```
All timing results are automatically written as structured JSON files into:
- `bench/results/cpp-serial/siam-i5-1340p/`
- `bench/results/cpp-openmp/siam-i5-1340p/`

---

## 2. Folder-by-Folder Architecture Explanation

```
CSE-402-Nuremical-Project/
├── backends/             <-- Independent simulation backends
│   ├── openmp/           <-- SIAM'S WORK: C++ Serial & OpenMP implementation
│   └── python_ref/       <-- Reference Python/NumPy baseline & ground truth
├── common/               <-- Shared Contract v1 files (interfaces & I/O)
│   ├── CONTRACT.md       <-- The rulebook for units, CLI flags, JSON format
│   ├── cpp/nbody_io.hpp  <-- Header-only C++ I/O library (parse_args, CSV, JSON)
│   └── python/           <-- Python I/O (nbio.py) & Initial Conditions exporter
├── data/                 <-- Simulation datasets
│   ├── ic/               <-- Initial conditions CSVs (N=100, N=1000, etc.)
│   └── ref/              <-- Ground-truth reference states for the gate
├── bench/                <-- Benchmarking & validation tools
│   ├── validate.py       <-- Validates position/velocity accuracy against ref
│   ├── check_result.py   <-- Validates result JSON schema v1
│   └── results/          <-- Raw benchmark JSON timings grouped by backend & machine
├── nbody_sim/            <-- Original foundational numerical simulation
│   ├── constants.py      <-- Gaussian astronomical units (AU, day, Msun)
│   ├── orbital_elements.py<-- Kepler equation solver & Asteroid Belt generator
│   ├── simulation.py     <-- Symplectic Velocity Verlet integrator
│   ├── onepn.py          <-- Einstein's 1st-order Post-Newtonian (1PN) correction
│   └── visualize.py      <-- Interactive 2D Matplotlib orbit visualizer
└── docs/                 <-- Team master guides & individual implementation guides
```

### Detailed Breakdown of Each Directory:

### `backends/openmp/` (Siam's Core Deliverable)
Contains the compiled C++ implementation.
* **`nbody_omp.cpp`**: Single source file compiled twice:
  - Without OpenMP $\rightarrow$ `nbody_serial.exe` (backend `cpp-serial`).
  - With OpenMP $\rightarrow$ `nbody_omp.exe` (backend `cpp-openmp`).
* **`CMakeLists.txt`**: Standard CMake build configuration.
* **`build_mingw.bat`**: One-click MinGW compilation script for Windows.
* **`test_m1_gate.py`**: Automated verification test against the reference data.
* **`sweep.py`**: Benchmark runner testing threads ($1, 2, 4, 8, 12, 16, 24$) and model sizes ($N=100$ to $10,000$).
* **`README.md`**: Hardware specification and OpenMP report documentation.

### `common/` (Contract v1)
Contains shared code ensuring all backends (Serial, OpenMP, MPI, CUDA) communicate through an identical interface:
* **`CONTRACT.md`**: Fixed standards: $G = 2.9591220828559115 \times 10^{-4}\text{ AU}^3 M_\odot^{-1}\text{ day}^{-2}$, time step $\Delta t = 0.1\text{ day}$, float64 precision.
* **`cpp/nbody_io.hpp`**: Fast C++ header-only parser and serializer. Reads initial conditions into Structure-of-Arrays format.
* **`python/nbio.py`**: Python reader/writer for CSV and JSON results.
* **`python/export_ic.py`**: Builds reproducible initial condition files for any $N$.

### `data/` (Inputs and Golden Outputs)
* **`data/ic/`**: Initial conditions CSV files (`ic_N100_s42.csv`, `ic_N1000_s42.csv`). Contains 3D coordinates $(x, y, z)$ and velocities $(v_x, v_y, v_z)$ for the Sun, 8 planets, Moon, and thousands of main-belt asteroids.
* **`data/ref/`**: Reference final states after 10 steps generated by the Python reference backend (`final_N100_steps10.csv`). Used by `validate.py` to ensure every C++ variant is 100% physically accurate.

### `bench/` (Validation & Benchmarking)
* **`validate.py`**: Checks that a backend's final state differs from the reference by less than $10^{-10}$ relative error.
* **`check_result.py`**: Verifies that generated benchmark JSONs match the exact Contract v1 schema.
* **`results/`**: Real benchmark timing runs. Organised by `<backend>/<machine_tag>/`. For Siam's machine, results reside in `bench/results/cpp-serial/siam-i5-1340p/` and `bench/results/cpp-openmp/siam-i5-1340p/`.

### `nbody_sim/` (The Numerical Foundation)
* **`constants.py`**: Unit definitions. Converts meters and kg into Astronomical Units ($\text{AU}$), days, and Solar Masses ($M_\odot$).
* **`orbital_elements.py`**: Solves Kepler's transcendental equation ($M = E - e \sin E$) via Newton-Raphson to position bodies in 3D space, and generates the synthetic Main Asteroid Belt ($a \in [2.1, 3.3]\text{ AU}$).
* **`simulation.py`**: Vectorized NumPy acceleration calculations and Velocity Verlet integrator.
* **`onepn.py` & `run_mercury_precession_test.py`**: Validates Einstein's General Relativity perihelion precession of Mercury ($42.98\text{ arcsec/century}$).
* **`visualize.py` & `web_demo/`**: Matplotlib GUI and HTML5 Canvas interactive browser simulation.

---

## 3. The Approach: How It Works & Why It Works

### 3.1 The Physical & Numerical Model
The gravitational force on body $p$ from all other $N-1$ bodies is:
$$\mathbf{a}_p = \sum_{j \neq p}^{N-1} G m_j \frac{\mathbf{r}_j - \mathbf{r}_p}{\left(\|\mathbf{r}_j - \mathbf{r}_p\|^2 + \epsilon^2\right)^{3/2}}$$

To step the simulation forward in time, we use the **Second-Order Velocity Verlet Integrator**:
1. **Position Update (Drift):**
   $$\mathbf{r}_{k+1} = \mathbf{r}_k + \mathbf{v}_k \Delta t + \frac{1}{2} \mathbf{a}_k \Delta t^2$$
2. **Force Evaluation:**
   Compute new accelerations $\mathbf{a}_{k+1}$ from the updated positions $\mathbf{r}_{k+1}$.
3. **Velocity Update (Kick):**
   $$\mathbf{v}_{k+1} = \mathbf{v}_k + \frac{1}{2} (\mathbf{a}_k + \mathbf{a}_{k+1}) \Delta t$$

#### Why Velocity Verlet?
- **Symplectic:** In Hamiltonian mechanics, Verlet preserves phase space volume. Total energy oscillates within a tight, bounded band rather than drifting outward or inward over time (unlike Euler or standard RK4).
- **Computational Cost:** Requires only **1 force calculation per time step** (compared to 4 evaluations per step in RK4), making it $4\times$ faster.

---

### 3.2 High-Performance C++ Design (Why It Is Fast)

#### 1. Structure-of-Arrays (SoA) Memory Layout
Instead of storing bodies as an array of structs (`struct Body { double x, y, z, m; } bodies[N]`), we use **Structure-of-Arrays**:
```cpp
std::vector<double> x, y, z, m;
```
*Why it works:* When calculating gravity, the processor needs continuous streams of $x$, $y$, and $z$ coordinates. Storing them in contiguous memory allows the CPU hardware prefetcher to load data directly into L1/L2 cache lines with zero wasted bandwidth.

#### 2. Zero Dynamic Allocations in Time Loop
Memory allocation (`malloc` or `new`) inside a simulation loop destroys performance. 
In `nbody_omp.cpp`, acceleration vectors (`ax`, `ay`, `az`, `bx`, `by`, `bz`) are allocated **once** during setup. At the end of each step, buffers are swapped using `std::swap` ($\mathcal{O}(1)$ pointer swap).

---

### 3.3 The 4 Algorithmic Variants Explained

#### Variant 1: `static` (Baseline Work-Sharing)
- **How it works:** Divides the $N$ target bodies into equal contiguous chunks among threads:
  `#pragma omp parallel for schedule(static)`
- **Why it works:** Thread 0 handles bodies $0 \dots \frac{N}{P}-1$, Thread 1 handles $\frac{N}{P} \dots \frac{2N}{P}-1$, etc. Positions are read-only, so there are **no locks and no data races**.

#### Variant 2: `dynamic` (Hybrid Architecture Balancer)
- **How it works:** Work is dynamically handed out in small blocks of 16 bodies:
  `#pragma omp parallel for schedule(dynamic, 16)`
- **Why it works:** Modern Intel CPUs (like your 13th Gen i5-1340P) have **4 fast Performance cores (P-cores)** and **8 slower Efficient cores (E-cores)**. Under a `static` schedule, the P-cores finish early and sit idle waiting for the E-cores to catch up! Under `dynamic`, the P-cores consume work chunks faster, eliminating idle wait time and achieving superior performance.

#### Variant 3: `simd` (Branch-Free Vectorization)
- **How it works:** In the standard loop, the computer checks `if (j == p) continue;`. This branch prevents CPU vectorization registers (AVX2). In `simd`, the loop is split into two branch-free loops:
  - Loop 1: $j \in [0, p)$
  - Loop 2: $j \in [p+1, N)$
  decorated with `#pragma omp simd reduction(+ : ax, ay, az)`.
- **Why it works:** The compiler can issue SIMD instructions that compute 4 double-precision interactions simultaneously per core.

#### Variant 4: `newton3` (Newton's Third Law Pairwise Summation)
- **How it works:** Newton's third law states $\mathbf{F}_{ij} = -\mathbf{F}_{ji}$. Instead of calculating all $N^2$ pairs, we only calculate pairs for $j > p$ ($\approx \frac{N^2}{2}$ interactions, cutting floating-point math by **50%**).
- **Why it works:** Each thread accumulates forces into thread-private buffers, and reduces them at the end. At $N=1,000$, this delivered a **$5.8\times$ speedup** on your machine!

---

### 3.4 Amdahl's Law & Small-$N$ Thread Overhead (Zhu Fig. 4 Reproduced)

During benchmark testing on your machine (`siam-i5-1340p`), we observed:
* **At $N = 1000$:** Adding threads speeds up execution from $15.0\text{ ms/step}$ down to $2.6\text{ ms/step}$ (**$5.8\times$ faster**).
* **At $N = 100$:**
  - 1 thread: $0.17\text{ ms/step}$.
  - 4 threads: $0.35\text{ ms/step}$ ($2\times$ slower).
  - 16 threads: $0.70\text{ ms/step}$ (**$4\times$ slower!**).

*Why this happens:* For small $N$, the actual physics calculation takes only microseconds. The overhead of launching threads, synchronizing barriers, and managing thread pools takes longer than the math itself! This empirical finding directly replicates **Figure 4 of Tailin Zhu (2020)**.
