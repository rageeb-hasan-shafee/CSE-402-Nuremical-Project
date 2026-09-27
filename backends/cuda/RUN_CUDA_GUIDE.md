# CUDA N-Body Simulation — Complete Execution & Benchmark Guide

**Author:** Shadhin  
**Project:** GPU-Accelerated N-Body Simulation of the Solar System (CSE-402)  
**Module:** `backends/cuda`  
**Target Hardware:** NVIDIA GeForce GTX 1660 SUPER (`sm_75`, 6 GB VRAM) & Kaggle Pascal P100 (`sm_60`)

---

## 1. Quickstart: Compiling & Running Locally on Windows

Your local machine is fully configured with CUDA 13.x and Visual Studio Build Tools 2022.

### Step 1: Open 64-Bit Developer Environment
In your PowerShell terminal, load the 64-bit native MSVC compiler:
```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64
```
*(Verify by running `cl` and `nvcc --version`).*

---

### Step 2: Compile Both Binaries

#### 1. Compile CUDA GPU Binary (`nbody_cuda.exe`)
From the `backends\cuda` directory:
```powershell
cd D:\4-1\sess\num\CSE-402-Nuremical-Project\backends\cuda
nvcc -O3 -std=c++17 -arch=sm_75 -I..\..\common\cpp nbody_cuda.cu -o nbody_cuda.exe
```

#### 2. Compile Scalar CPU Binary (`nbody_serial.exe`)
From the `backends\serial` directory:
```powershell
cd D:\4-1\sess\num\CSE-402-Nuremical-Project\backends\serial
cl /O2 /std:c++17 /EHsc /I..\..\common\cpp nbody_serial.cpp /Fe:nbody_serial.exe
```

---

### Step 3: Run a Single Simulation

Run 100 Velocity Verlet steps on $N=1,000$ bodies:
```powershell
cd D:\4-1\sess\num\CSE-402-Nuremical-Project

# CUDA Tiled Double Precision (FP64)
.\backends\cuda\nbody_cuda.exe --input data\ic\ic_N1000_s42.csv --steps 100 --variant tiled-f64

# CUDA Tiled Single Precision (FP32)
.\backends\cuda\nbody_cuda.exe --input data\ic\ic_N1000_s42.csv --steps 100 --variant tiled-f32
```

---

### Step 4: Verify Numerical Accuracy Against CPU Baseline

Run both backends and verify that orbital states match within floating-point tolerance:
```powershell
# 1. Run CPU Baseline
.\backends\serial\nbody_serial.exe --input data\ic\ic_N1000_s42.csv --steps 100 --final-state backends\serial\out\serial_1000.csv --output-json bench\results\cpp-serial\local\serial_1000.json

# 2. Run CUDA GPU
.\backends\cuda\nbody_cuda.exe --input data\ic\ic_N1000_s42.csv --steps 100 --variant tiled-f64 --final-state backends\cuda\out\cuda_1000.csv --output-json bench\results\cuda\shadhin-gtx1660s\cuda_1000.json --machine shadhin-gtx1660s

# 3. Compare with tolerance 1e-9
python bench\compare.py --serial-json bench\results\cpp-serial\local\serial_1000.json --gpu-json bench\results\cuda\shadhin-gtx1660s\cuda_1000.json --serial-csv backends\serial\out\serial_1000.csv --gpu-csv backends\cuda\out\cuda_1000.csv --tol 1e-9
```

Output:
```text
========================================================
 NUMERICAL ACCURACY VERIFICATION
 Reference (Scalar CPU): backends\serial\out\serial_1000.csv
 Test (CUDA GPU):        backends\cuda\out\cuda_1000.csv
 Tolerance:              1.0e-09
========================================================
 Max Position Error: 4.440892e-16 AU  <-- Verified down to machine precision!
 RMS Position Error: 3.267239e-17 AU

 >>> RESULT: PASS (All states match within tolerance 1.0e-09) <<<
```

---

### Step 5: Run Full Automated Benchmark Sweep

To benchmark across all body sizes ($N=100 \dots 50{,}000$), all kernel variants, and block sizes:
```powershell
python backends\cuda\sweep_cuda.py --machine shadhin-gtx1660s --steps 100 --repeats 3
```
* Generates all missing initial condition files automatically.
* Benchmarks the scalar CPU baseline for every $N$.
* Runs each CUDA configuration 3 times and takes the median.
* Automatically creates/updates the formatted Markdown report:
  `bench\results\cuda\shadhin-gtx1660s\BENCHMARK_REPORT.md`

---

## 2. Empirical Benchmark Results (NVIDIA GeForce GTX 1660 SUPER)

Measured live on your machine (100 Velocity Verlet steps, $\Delta t = 0.1$ days):

### Table 1: N Scaling & GPU Speed-Up over Scalar CPU
| $N$ Bodies | Scalar CPU (ms/step) | GTX 1660 SUPER FP64 (ms/step) | **FP64 Speed-Up** | GTX 1660 SUPER FP32 (ms/step) | **FP32 Speed-Up** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **100** | 0.053 ms | 0.174 ms | 0.31x | 0.018 ms | **2.99x** |
| **500** | 1.023 ms | 0.814 ms | 1.26x | 0.049 ms | **20.71x** |
| **1,000** | 4.085 ms | 1.534 ms | **2.66x** | 0.076 ms | **53.45x** |
| **2,000** | 16.238 ms | 2.663 ms | **6.10x** | 0.145 ms | **112.14x** |
| **5,000** | 103.833 ms | 7.002 ms | **14.83x** | 0.375 ms | **277.11x** |
| **10,000** | 411.105 ms | 25.660 ms | **16.02x** | 0.879 ms | **467.68x** |
| **20,000** | 1,669.400 ms | 102.273 ms | **16.32x** | 3.261 ms | **511.90x** |
| **50,000** | 10,223.800 ms | 582.468 ms | **17.55x** | 16.992 ms | **601.67x** |

---

### Table 2: Memory Hierarchy Optimization (Naive vs. Tiled Shared Memory)
| $N$ Bodies | Naive FP64 (ms/step) | Tiled FP64 (ms/step) | Naive FP32 (ms/step) | Tiled FP32 (ms/step) |
|:---:|:---:|:---:|:---:|:---:|
| **1,000** | 1.332 ms | 1.534 ms | 0.059 ms | 0.076 ms |
| **5,000** | 6.378 ms | 7.002 ms | 0.278 ms | 0.375 ms |
| **10,000** | 25.989 ms | 25.660 ms | 0.812 ms | 0.879 ms |

---

### Table 3: Block Size Sensitivity ($N = 10{,}000$)
Demonstrates hardware occupancy and register allocation:

| Threads per Block ($B$) | Loop Time (100 steps) | Time per Step | Notes |
|:---:|:---:|:---:|---|
| **64** | 2.542 s | 25.420 ms/step | Good occupancy |
| **128** | 2.542 s | 25.423 ms/step | Near optimal |
| **256** | 2.566 s | 25.660 ms/step | Default standard block size |
| **512** | 2.556 s | 25.566 ms/step | High thread occupancy |
| **1024** | 5.112 s | 51.122 ms/step | **$2\times$ slowdown** (register pressure limits active warps) |

---

## 3. Cloud Extension: Kaggle Notebooks (NVIDIA Pascal P100)

To compare your local Turing GTX 1660 SUPER against enterprise Pascal architecture where FP64 runs at **1/2 rate**:

1. Go to **[kaggle.com/code](https://www.kaggle.com/code)** $\to$ New Notebook $\to$ Select **GPU P100** under Accelerator.
2. Run the cells:

```python
# 1. Clone & Checkout
!git clone https://github.com/rageeb-hasan-shafee/CSE-402-Nuremical-Project.git
%cd CSE-402-Nuremical-Project
!git checkout shadhin_dev

# 2. Compile for Pascal P100 (sm_60)
%cd backends/cuda
!nvcc -O3 -std=c++17 -arch=sm_60 -I../../common/cpp nbody_cuda.cu -o nbody_cuda
%cd ../..

# 3. Run Benchmark Sweep on P100
!python backends/cuda/sweep_cuda.py --machine shadhin-kaggle-p100 --steps 100 --repeats 3

# 4. Download Results ZIP
!zip -qr kaggle_p100_results.zip bench/results/cuda/shadhin-kaggle-p100
from IPython.display import FileLink
FileLink("kaggle_p100_results.zip")
```

---

## 4. Key Takeaways for the Final Project Report

1. **Massive Parallel Scaling:** 
   At $N = 10,000$, the CUDA implementation speeds up the simulation by **$16.02\times$ in double precision (FP64)** and **$467.68\times$ in single precision (FP32)** over the single-core CPU baseline.
2. **Small $N$ Kernel Launch Overhead:**
   At $N = 100$, the CPU baseline is slightly faster (0.053 ms vs 0.174 ms). This demonstrates the classic GPU trade-off: PCIe launch latency and under-utilized Streaming Multiprocessors make GPUs inefficient for tiny workloads.
3. **Hardware Architecture Trait (FP32 vs FP64):**
   On your GTX 1660 SUPER (Turing architecture), FP32 runs **$29\times$ faster** than FP64 ($0.879$ ms vs $25.66$ ms at $N=10,000$), perfectly illustrating the consumer 1/32 FP64 ALU hardware limitation.
4. **Occupancy Limit at $B=1024$:**
   Setting block size to 1024 threads causes execution time to double (from 25.5 ms to 51.1 ms), showing that exceeding optimal register limits restricts simultaneous active warps per SM.
