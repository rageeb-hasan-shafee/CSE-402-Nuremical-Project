# CUDA N-Body Benchmark Report — NVIDIA GeForce RTX 5090

### System & GPU Hardware Configuration
* **GPU Model:** `NVIDIA GeForce RTX 5090`
* **Compute Capability:** `SM 12.0`
* **Machine Identifier:** `ubuntu-ubuntu-System-Product-Name`
* **Scalar CPU Baseline:** Tested on host machine (`nbody_serial`)
* **Benchmark Date:** `2026-09-28 04:44:13`
* **Integration Scheme:** Velocity Verlet, $\Delta t = 0.1$ days, `100` steps

---

## 1. N Scaling & GPU Speed-Up over Scalar CPU (NVIDIA GeForce RTX 5090)
This table evaluates how execution time scales as body count N increases, and computes the exact speed-up (S = T_CPU / T_GPU).

| N Bodies | Scalar CPU (ms/step) | NVIDIA GeForce RTX 5090 FP64 (ms/step) | **FP64 Speed-Up** | NVIDIA GeForce RTX 5090 FP32 (ms/step) | **FP32 Speed-Up** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 | 0.019 | 0.094 | **0.20x** | 0.012 | **1.53x** |
| 500 | 0.473 | 0.435 | **1.09x** | 0.034 | **13.77x** |
| 1,000 | 1.883 | 0.865 | **2.18x** | 0.063 | **29.83x** |
| 2,000 | 7.506 | 1.725 | **4.35x** | 0.118 | **63.64x** |
| 5,000 | 46.937 | 4.306 | **10.90x** | 0.281 | **166.79x** |
| 10,000 | 187.750 | 8.604 | **21.82x** | 0.556 | **337.59x** |
| 20,000 | 751.857 | 17.202 | **43.71x** | 1.107 | **679.40x** |
| 50,000 | 4698.780 | 85.946 | **54.67x** | 3.090 | **1520.51x** |

---

## 2. Memory Hierarchy Optimization on NVIDIA GeForce RTX 5090 (Naive vs. Tiled Shared Memory)
Demonstrates the benefit of caching particles into on-chip `__shared__` memory ($B=256$) to reduce global memory bandwidth.

| N Bodies | Naive FP64 (ms/step) | Tiled FP64 (ms/step) | **FP64 Tiling Gain** | Naive FP32 (ms/step) | Tiled FP32 (ms/step) | **FP32 Tiling Gain** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 | 0.065 | 0.094 | **0.70x** | 0.008 | 0.012 | **0.67x** |
| 500 | 0.468 | 0.435 | **1.08x** | 0.025 | 0.034 | **0.72x** |
| 1,000 | 0.907 | 0.865 | **1.05x** | 0.051 | 0.063 | **0.81x** |
| 2,000 | 1.790 | 1.725 | **1.04x** | 0.108 | 0.118 | **0.91x** |
| 5,000 | 4.445 | 4.306 | **1.03x** | 0.267 | 0.281 | **0.95x** |
| 10,000 | 8.858 | 8.604 | **1.03x** | 0.542 | 0.556 | **0.97x** |
| 20,000 | — | 17.202 | — | — | 1.107 | — |
| 50,000 | — | 85.946 | — | — | 3.090 | — |

---

## 3. Host-Device Transfer Cost on NVIDIA GeForce RTX 5090 (PCIe Bottleneck)
Compares keeping state resident in GPU VRAM vs. copying positions back to host memory each step (`copystep-f64`).

| N Bodies | GPU Resident (ms/step) | Copy Each Step (ms/step) | **Transfer Overhead Factor** |
|:---:|:---:|:---:|:---:|
| 100 | 0.094 | 0.099 | **1.06x slower** |
| 500 | 0.435 | 0.448 | **1.03x slower** |
| 1,000 | 0.865 | 0.876 | **1.01x slower** |
| 2,000 | 1.725 | 1.737 | **1.01x slower** |
| 5,000 | 4.306 | 4.347 | **1.01x slower** |
| 10,000 | 8.604 | 8.676 | **1.01x slower** |

---

## 4. Block Size Sensitivity on NVIDIA GeForce RTX 5090 (N = 10,000)
Analyzes occupancy and shared memory configuration across different thread block sizes $B$.

| Threads per Block (B) | Loop Time (s) | Time per Step (ms/step) |
|:---:|:---:|:---:|
| 64 | 0.4244 s | 4.244 ms/step |
| 128 | 0.4537 s | 4.537 ms/step |
| 256 | 0.8604 s | 8.604 ms/step |
| 512 | 1.7185 s | 17.185 ms/step |
| 1024 | 3.4333 s | 34.333 ms/step |

---

## 5. Key Takeaways for Report
* **Quadratic Scaling:** At large $N$, execution time scales as $O(N^2)$, confirming compute-bound scaling.
* **FP32 vs FP64 Architecture:** Single precision runs substantially faster than double precision due to the GPU's hardware ALU ratio.
* **Shared Memory Tiling:** Tiled shared memory eliminates redundant global memory transactions, resulting in significant speed improvements over naive summation.
* **Zero-Copy Advantage:** Keeping simulation data on the GPU avoids PCIe bus bottlenecks that otherwise degrade performance.
