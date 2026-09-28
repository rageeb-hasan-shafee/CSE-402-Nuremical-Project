# OpenMP Parallel Benchmark Report — Siam

**Author:** Siam  
**Backend:** `cpp-openmp` & `cpp-serial`  
**Machine Tag:** `siam-i5-1340p`  
**Hardware Specifications:**
- **CPU:** 13th Gen Intel® Core™ i5-1340P (12 Cores / 16 Threads: 4 Performance-cores @ 4.60 GHz + 8 Efficient-cores @ 3.40 GHz)
- **Compiler:** MinGW-w64 GCC 8.1.0 (`x86_64-w64-mingw32`, posix threads, SEH)
- **Optimization Flags:** `-O3 -mavx2 -mfma -fopenmp -std=c++17`
- **Operating System:** Microsoft Windows 11 Home (x86_64)

---

## 1. Executive Summary & Deliverables Status

| Requirement / Deliverable | Status | Details |
|---|---|---|
| **Contract v1 Compliance** | **PASS** | Validated via `bench/check_result.py` |
| **M1 Correctness Gate** | **PASS (10/10)** | Relative position & velocity error $< 2 \times 10^{-16}$ against reference |
| **All 4 OpenMP Variants** | **PASS** | `static`, `dynamic`, `simd`, `newton3` implemented & verified |
| **Hardware Benchmark Matrix** | **COMPLETE** | 346 JSON benchmark runs committed under `bench/results/` |
| **Problem Size Sweep ($N$)** | **COMPLETE** | $N \in [100, 500, 1000, 2000, 5000, 10000]$ |
| **Thread Scaling ($P$)** | **COMPLETE** | $P \in [1, 2, 4, 8, 12, 16, 24]$ |

---

## 2. Key Experimental Findings

### 2.1. Maximum Parallel Speedup ($N = 10,000$)
At large $N = 10,000$, where compute work ($O(N^2) = 100 \times 10^6$ interactions per step) completely dominates thread synchronization overhead:
- **`cpp-serial` (Single-core baseline):** $1489.6\text{ ms/step}$ ($1.0\times$ anchor)
- **`cpp-openmp (static, 2 threads):`** $742.5\text{ ms/step}$ (**$2.01\times$ — Perfect linear scaling**)
- **`cpp-openmp (static, 4 threads):`** $400.9\text{ ms/step}$ (**$3.72\times$ — Near-linear across 4 P-cores**)
- **`cpp-openmp (static, 8 threads):`** $204.8\text{ ms/step}$ (**$7.27\times$ scaling**)
- **`cpp-openmp (static, 16 threads):`** $109.9\text{ ms/step}$ (**$13.55\times$ overall speedup!**)
- **`cpp-openmp (static, 24 threads):`** $113.4\text{ ms/step}$ (Slows down due to hyperthread oversubscription context switching)

---

### 2.2. Intel Hybrid Architecture Effect (P-cores vs. E-cores)
The Intel Core i5-1340P has 4 Performance-cores (high IPC, high clock) and 8 Efficient-cores (lower IPC, lower clock).
- **`static` scheduling:** Divides bodies into equal blocks ($N/P$). The fastest threads on P-cores finish early and wait idly at the OpenMP barrier for the slower E-core threads to finish.
- **`dynamic` scheduling:** Delivers work in small dynamic chunks (`schedule(dynamic, 64)`). The fast P-cores consume more chunks while E-cores process fewer chunks, naturally balancing the load across asymmetric cores.
- **Result at $N=5000, P=16$:**
  - `static`: $44.2\text{ ms/step}$
  - `dynamic`: **$30.5\text{ ms/step}$ ($1.45\times$ faster than static!)**

---

### 2.3. Newton's 3rd Law Optimization (`newton3`)
By exploiting $\vec{F}_{ji} = -\vec{F}_{ij}$, the number of pair force evaluations is halved from $N(N-1)$ to $\frac{N(N-1)}{2}$:
- At $N=5000$, single-thread runtime drops from $375.7\text{ ms}$ (`static`) to **$195.5\text{ ms}$ (`newton3`)** — an exact **$1.92\times$ reduction in computational effort**.
- At $N=5000$ with 16 threads, `newton3` achieves **$24.48\text{ ms/step}$**, achieving the highest throughput across the entire project (**over $1.02 \times 10^9$ pairs/sec**).

---

### 2.4. Small-$N$ Thread Overhead (Zhu 2020 Fig. 4 Reproduction)
At small $N = 100$ (where work per step is only $10^4$ pairs), thread creation, fork/join barriers, and cache migration overhead exceed the arithmetic payload:
- **1 thread:** $0.168\text{ ms/step}$
- **4 threads:** $0.336\text{ ms/step}$ ($2\times$ slower)
- **16 threads:** $0.704\text{ ms/step}$ (**$4.2\times$ slower**)

> **Theoretical Insight:** This directly confirms Zhu (2020) Section 4.2: for small system sizes, serial execution is optimal; parallelization should only be engaged once $N \ge 500$.

---

## 3. Comprehensive Timing Summary Table (ms/step)

| Problem Size ($N$) | `cpp-serial` | `omp-static` ($P=1$) | `omp-static` ($P=4$) | `omp-static` ($P=16$) | `omp-dynamic` ($P=16$) | `omp-newton3` ($P=16$) | Max Speedup |
|---|---|---|---|---|---|---|---|
| **$N = 100$** | **0.17 ms** | 0.17 ms | 0.34 ms | 0.70 ms | 0.63 ms | 1.14 ms | **1.0× (Serial wins)** |
| **$N = 500$** | 3.75 ms | 3.80 ms | 1.61 ms | 1.59 ms | 1.40 ms | **0.95 ms** | **3.9×** |
| **$N = 1,000$** | 14.92 ms | 15.01 ms | 6.18 ms | 4.75 ms | 3.88 ms | **2.61 ms** | **5.7×** |
| **$N = 2,000$** | 59.85 ms | 60.10 ms | 18.20 ms | 16.10 ms | 14.20 ms | **9.85 ms** | **6.1×** |
| **$N = 5,000$** | 374.5 ms | 375.7 ms | 104.7 ms | 44.20 ms | 30.55 ms | **24.48 ms** | **15.3×** |
| **$N = 10,000$** | 1489.6 ms | 1475.1 ms | 400.9 ms | 109.9 ms | 88.40 ms | **68.20 ms** | **21.8×** |

---

## 4. Hardware Profile for Fahad (`bench/machines.json`)

```json
{
  "machine": "siam-i5-1340p",
  "owner": "Siam",
  "cpu": {
    "model": "13th Gen Intel(R) Core(TM) i5-1340P",
    "architecture": "x86_64 (Raptor Lake)",
    "cores_physical": 12,
    "cores_logical": 16,
    "p_cores": 4,
    "e_cores": 8,
    "base_clock_ghz": 1.90,
    "boost_clock_ghz": 4.60
  },
  "ram_gb": 16.0,
  "os": "Microsoft Windows 11 Home 64-bit",
  "compiler": {
    "name": "MinGW-w64 GCC",
    "version": "8.1.0",
    "flags": "-O3 -mavx2 -mfma -fopenmp -std=c++17"
  }
}
```
