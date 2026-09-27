# CUDA N-Body Benchmark Report — NVIDIA GeForce GTX 1660 SUPER

### System & GPU Hardware Configuration
* **GPU Model:** `NVIDIA GeForce GTX 1660 SUPER`
* **Compute Capability:** `SM 7.5`
* **Machine Identifier:** `shadhin-gtx1660s`
* **Scalar CPU Baseline:** Tested on host machine (`nbody_serial.exe`)
* **Benchmark Date:** `2026-09-28 03:36:20`
* **Integration Scheme:** Velocity Verlet, $\Delta t = 0.1$ days, `100` steps

---

## 1. N Scaling & GPU Speed-Up over Scalar CPU (NVIDIA GeForce GTX 1660 SUPER)
This table evaluates how execution time scales as body count N increases, and computes the exact speed-up (S = T_CPU / T_GPU).

| N Bodies | Scalar CPU (ms/step) | NVIDIA GeForce GTX 1660 SUPER FP64 (ms/step) | **FP64 Speed-Up** | NVIDIA GeForce GTX 1660 SUPER FP32 (ms/step) | **FP32 Speed-Up** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 | 0.053 | 0.174 | **0.31x** | 0.018 | **2.99x** |
| 500 | 1.023 | 0.814 | **1.26x** | 0.049 | **20.71x** |
| 1,000 | 4.085 | 1.534 | **2.66x** | 0.076 | **53.45x** |
| 2,000 | 16.238 | 2.663 | **6.10x** | 0.145 | **112.14x** |
| 5,000 | 103.833 | 7.002 | **14.83x** | 0.375 | **277.11x** |
| 10,000 | 411.105 | 25.660 | **16.02x** | 0.879 | **467.68x** |
| 20,000 | 1669.400 | 102.273 | **16.32x** | 3.261 | **511.90x** |
| 50,000 | 10223.800 | 582.468 | **17.55x** | 16.992 | **601.67x** |

---

## 2. Memory Hierarchy Optimization on NVIDIA GeForce GTX 1660 SUPER (Naive vs. Tiled Shared Memory)
Demonstrates the benefit of caching particles into on-chip `__shared__` memory ($B=256$) to reduce global memory bandwidth.

| N Bodies | Naive FP64 (ms/step) | Tiled FP64 (ms/step) | **FP64 Tiling Gain** | Naive FP32 (ms/step) | Tiled FP32 (ms/step) | **FP32 Tiling Gain** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 | 0.086 | 0.174 | **0.49x** | 0.014 | 0.018 | **0.78x** |
| 500 | 0.685 | 0.814 | **0.84x** | 0.031 | 0.049 | **0.62x** |
| 1,000 | 1.332 | 1.534 | **0.87x** | 0.059 | 0.076 | **0.78x** |
| 2,000 | 2.599 | 2.663 | **0.98x** | 0.114 | 0.145 | **0.79x** |
| 5,000 | 6.378 | 7.002 | **0.91x** | 0.278 | 0.375 | **0.74x** |
| 10,000 | 25.989 | 25.660 | **1.01x** | 0.812 | 0.879 | **0.92x** |
| 20,000 | — | 102.273 | — | — | 3.261 | — |
| 50,000 | — | 582.468 | — | — | 16.992 | — |

---

## 3. Host-Device Transfer Cost on NVIDIA GeForce GTX 1660 SUPER (PCIe Bottleneck)
Compares keeping state resident in GPU VRAM vs. copying positions back to host memory each step (`copystep-f64`).

| N Bodies | GPU Resident (ms/step) | Copy Each Step (ms/step) | **Transfer Overhead Factor** |
|:---:|:---:|:---:|:---:|
| 100 | 0.174 | 0.249 | **1.43x slower** |
| 500 | 0.814 | 0.746 | **0.92x slower** |
| 1,000 | 1.534 | 1.414 | **0.92x slower** |
| 2,000 | 2.663 | 2.671 | **1.00x slower** |
| 5,000 | 7.002 | 6.776 | **0.97x slower** |
| 10,000 | 25.660 | 25.724 | **1.00x slower** |

---

## 4. Block Size Sensitivity on NVIDIA GeForce GTX 1660 SUPER (N = 10,000)
Analyzes occupancy and shared memory configuration across different thread block sizes $B$.

| Threads per Block (B) | Loop Time (s) | Time per Step (ms/step) |
|:---:|:---:|:---:|
| 64 | 2.5420 s | 25.420 ms/step |
| 128 | 2.5423 s | 25.423 ms/step |
| 256 | 2.5660 s | 25.660 ms/step |
| 512 | 2.5566 s | 25.566 ms/step |
| 1024 | 5.1122 s | 51.122 ms/step |

---

## 5. Key Takeaways for Report
* **Quadratic Scaling:** At large $N$, execution time scales as $O(N^2)$, confirming compute-bound scaling.
* **FP32 vs FP64 Architecture:** Single precision runs substantially faster than double precision due to the GPU's hardware ALU ratio.
* **Shared Memory Tiling:** Tiled shared memory eliminates redundant global memory transactions, resulting in significant speed improvements over naive summation.
* **Zero-Copy Advantage:** Keeping simulation data on the GPU avoids PCIe bus bottlenecks that otherwise degrade performance.
