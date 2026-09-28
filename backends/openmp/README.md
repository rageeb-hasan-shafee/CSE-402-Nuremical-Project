# OpenMP Backend & Compiled Serial Baseline (`backends/openmp`)

> **Owner:** Saidul Anam Siam (2105156)  
> **Course:** CSE 402 - Simulation and Modeling  
> **Contract:** Contract v1 (`common/CONTRACT.md`)  
> **Workload:** OpenMP (Shared Memory C++) + Compiled C++ Serial Baseline (`cpp-serial`)

---

## 1. Overview

This module provides two high-performance C++ executables compiled from `nbody_omp.cpp`:
1. **`nbody_serial` (`cpp-serial`):** The compiled serial CPU baseline against which all parallel backends (OpenMP, MPI, CUDA) measure their absolute speedup ($S = T_{\text{serial}} / T_{\text{parallel}}$).
2. **`nbody_omp` (`cpp-openmp`):** Multi-threaded shared-memory implementation utilizing OpenMP work-sharing constructs.

Both executables use **identical `-O3 -std=c++17` optimization flags** to ensure fair baseline comparisons.

---

## 2. Hardware Specification (`--machine siam-i5-1340p`)

| Component | Specification |
|---|---|
| **CPU Model** | 13th Gen Intel(R) Core(TM) i5-1340P |
| **Physical Architecture** | Hybrid architecture: 4 Performance cores (P-cores) + 8 Efficient cores (E-cores) |
| **Cores / Threads** | 12 physical cores / 16 logical hardware threads |
| **Max Turbo Frequency** | Up to 4.60 GHz |
| **System RAM** | 16 GB RAM |
| **Operating System** | Windows 11 Home 64-bit |
| **Compiler** | MinGW-W64 GCC 8.1.0 (`Target: x86_64-w64-mingw32`, POSIX threads, OpenMP 4.5) |

---

## 3. Algorithmic Variants

Select variants using `--variant <name>`:

| Variant | Description | Key Performance Trade-off |
|---|---|---|
| **`static`** *(default)* | `#pragma omp parallel for schedule(static)` over targets $p$, full inner loop over sources $j$. | Optimal for homogeneous workloads; can suffer from load imbalance on hybrid P/E CPUs. |
| **`dynamic`** | `#pragma omp parallel for schedule(dynamic, 16)`. | Dynamically dispatches chunks of 16 bodies. Critical on hybrid architectures (prevents slow E-cores from causing tail latency). |
| **`simd`** | Splits inner loop into `[0, p)` and `[p+1, n)` to eliminate the branch (`j == p`), annotated with `#pragma omp simd reduction(+ : ax, ay, az)`. | Unlocks CPU vector registers (AVX2/FMA) for up to $2\times$ faster single-core compute. |
| **`newton3`** | Uses Newton's 3rd Law ($\mathbf{a}_{ij} = -\mathbf{a}_{ji}$) for $j > p$. Halves the arithmetic floating-point interactions. | Uses thread-private buffers reduced in a critical section. Wins when compute-bound, loses if memory reduction overhead dominates. |

---

## 4. Build Instructions

### Option A: Windows MinGW (Quick Batch Script)
From the repository root:
```cmd
cd backends\openmp
build_mingw.bat
```
Executables are placed into `backends/openmp/build/`.

### Option B: CMake (Cross-Platform)
```bash
# From repo root
cmake -S backends/openmp -B backends/openmp/build -DCMAKE_BUILD_TYPE=Release
cmake --build backends/openmp/build --config Release
```
*(On macOS with Homebrew: pass `-DOpenMP_ROOT=$(brew --prefix libomp)`)*.

---

## 5. Running the Executables (Contract CLI)

### Running Serial Baseline
```bash
backends/openmp/build/nbody_serial --input data/ic/ic_N1000_s42.csv --steps 100 --dt 0.1 \
    --output-json bench/results/cpp-serial/siam-i5-1340p/cpp-serial_static_N1000_P1_r1.json \
    --final-state backends/openmp/build/final_serial_N1000.csv
```

### Running OpenMP Parallel Engine
```bash
backends/openmp/build/nbody_omp --input data/ic/ic_N1000_s42.csv --steps 100 --dt 0.1 \
    --workers 16 --variant dynamic \
    --output-json bench/results/cpp-openmp/siam-i5-1340p/cpp-openmp_dynamic_N1000_P16_r1.json \
    --final-state backends/openmp/build/final_omp_N1000.csv
```

---

## 6. Verification & Correctness Gate (M1 Gate)

Run the verification test against the reference solution:
```bash
python backends/openmp/test_m1_gate.py
```
**Pass Criteria:** Max relative position and velocity errors must be **$< 10^{-10}$** (Contract §C6).

---

## 7. Running Full Benchmark Sweeps (M2)

Run automated sweeps across body counts ($N$), thread counts ($P$), and variants:
```bash
# Quick sanity sweep (N = 100, 500, 1000)
python backends/openmp/sweep.py --quick

# Full production sweep (N = 100 to 10,000, threads 1 to 24)
python backends/openmp/sweep.py
```
Results are saved to:
- `bench/results/cpp-serial/siam-i5-1340p/*.json`
- `bench/results/cpp-openmp/siam-i5-1340p/*.json`
