# CUDA GPU N-body Simulation Backend

**Owner:** Shadhin  
**Module:** `backends/cuda`  
**Base Paper:** Zhu (2020), *N-body Simulations of the Solar System with CPU-based Parallel Methods*  

---

## 1. Overview

This backend implements GPU-accelerated gravitational $O(N^2)$ N-body simulation using NVIDIA CUDA. It provides the primary parallel computing extension over the CPU-based OpenMP/MPI implementations in the base paper.

### Key Features
* **Shared Memory Tiling (`tiled`)**: Threads in a 256-thread block cooperatively load body positions into `__shared__` memory, reducing global memory traffic by $B = 256\times$.
* **Zero Host-Device Transfer in Loop**: The full Velocity Verlet time-stepping (`drift` $\to$ `accel` $\to$ `kick`) remains resident in GPU VRAM across all steps.
* **Negative Control (`copystep`)**: Transfers positions back to the host every step to quantify the PCIe transfer latency bottleneck.
* **Precision Analysis (`f64` vs `f32`)**: Compares double-precision numerical stability against single-precision hardware throughput (especially on Tesla T4 where FP64 is 1/32 speed).

---

## 2. Directory Structure

```
backends/cuda/
├── nbody_cuda.cu      # Core CUDA source (naive, tiled, copystep; f64 & f32)
├── sweep_cuda.py      # Automated parameter sweep benchmark script
├── colab_run.ipynb    # Google Colab / Kaggle execution notebook
├── README.md          # This documentation
└── .gitignore         # Build and output ignores
```

---

## 3. Compilation

Use `nvcc` with C++17 and `-O3`:

```bash
# Tesla T4 (Google Colab / AWS G4dn)
nvcc -O3 -std=c++17 -arch=sm_75 -I../../common/cpp nbody_cuda.cu -o nbody_cuda

# Pascal P100 (Kaggle)
nvcc -O3 -std=c++17 -arch=sm_60 -I../../common/cpp nbody_cuda.cu -o nbody_cuda

# Ampere A100
nvcc -O3 -std=c++17 -arch=sm_80 -I../../common/cpp nbody_cuda.cu -o nbody_cuda

# Ada Lovelace L4
nvcc -O3 -std=c++17 -arch=sm_89 -I../../common/cpp nbody_cuda.cu -o nbody_cuda

# Local NVIDIA GPU (CUDA >= 12)
nvcc -O3 -std=c++17 -arch=native -I../../common/cpp nbody_cuda.cu -o nbody_cuda
```

---

## 4. CLI Arguments Contract

`nbody_cuda` accepts standard flags:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--input` | string | *required* | Path to initial conditions CSV (`name,mass,x,y,z,vx,vy,vz`) |
| `--steps` | int | `10` | Number of Velocity Verlet time steps |
| `--dt` | double | `0.1` | Time step size in days |
| `--variant` | string | `tiled-f64` | Format: `<kernel>-<precision>` (`naive-f64`, `tiled-f64`, `copystep-f64`, `tiled-f32`, `naive-f32`) |
| `--block-size` | int | `256` | CUDA block size (threads per block, between 32 and 1024) |
| `--final-state` | string | `""` | Output path to save final CSV state for validation |
| `--output-json` | string | `""` | Output path to write benchmark timing JSON |
| `--machine` | string | `"unknown"`| Machine tag (e.g. `shadhin-colab-t4`) |

### Example Run
```bash
./nbody_cuda \
    --input ../../data/ic/ic_N1000_s42.csv \
    --steps 100 \
    --dt 0.1 \
    --variant tiled-f64 \
    --block-size 256 \
    --final-state out/final_tiled_1000.csv \
    --output-json out/res_tiled_1000.json \
    --machine shadhin-colab-t4
```

---

## 5. Running the Sweeps

Execute the automated benchmark matrix:

```bash
python sweep_cuda.py --machine shadhin-colab-t4 --steps 100 --repeats 3
```

Results are saved into:
```
bench/results/cuda/<machine>/cuda_<variant>_B<block>_N<N>_P1_r<k>.json
```
