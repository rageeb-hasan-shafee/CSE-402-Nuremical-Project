# CUDA N-Body Simulation — Ubuntu Linux Benchmark Guide

This guide covers everything needed to compile, execute, validate, and benchmark the CUDA GPU and C++ Serial N-Body simulation on **Ubuntu Linux** (WSL2, Native Ubuntu 20.04/22.04/24.04, or Cloud VMs like AWS/GCP/Lambda).

---

## 1. Prerequisites & Environment Setup

### Step 1: Verify NVIDIA Driver
Ensure your NVIDIA GPU is recognized and proprietary drivers are installed:
```bash
nvidia-smi
```
*(You should see your GPU model, driver version, and CUDA version).*

### Step 2: Install Build Essentials and CUDA Toolkit
If `g++` or `nvcc` are not installed, install them via `apt`:
```bash
sudo apt update
sudo apt install -y build-essential nvidia-cuda-toolkit git python3 python3-pip python3-venv
```
Verify the compilers:
```bash
g++ --version
nvcc --version
```

### Step 3: Set Up Python Virtual Environment
From the repository root (`CSE-402-Nuremical-Project`):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas matplotlib
```

---

## 2. Identify Your GPU Architecture (`-arch=sm_XX`)

Check your GPU model:
```bash
nvidia-smi --query-gpu=name --format=csv,noheader
```

Match your GPU architecture flag for `nvcc`:

| GPU Architecture | Models | NVCC Flag |
|:---|:---|:---:|
| **Pascal** | GTX 1060 / 1070 / 1080 / Tesla P100 | `-arch=sm_60` |
| **Volta** | Titan V / Tesla V100 | `-arch=sm_70` |
| **Turing** | GTX 1650 / 1660 / RTX 2060 / 2070 / 2080 / Tesla T4 | `-arch=sm_75` |
| **Ampere** | RTX 3060 / 3070 / 3080 / 3090 / A10 / A100 | `-arch=sm_86` *(or `sm_80` for A100)* |
| **Ada Lovelace** | RTX 4060 / 4070 / 4080 / 4090 / L4 / L40 | `-arch=sm_89` |
| **Hopper** | H100 / H200 | `-arch=sm_90` |

*(Replace `-arch=sm_75` in the compilation commands below with your GPU's flag if different).*

---

## 3. Compile Both Binaries

### 1. Compile the CUDA GPU Binary (`nbody_cuda`)
```bash
cd backends/cuda
nvcc -O3 -std=c++17 -arch=sm_75 -I../../common/cpp nbody_cuda.cu -o nbody_cuda
chmod +x nbody_cuda
cd ../..
```

### 2. Compile the Single-Core C++ Serial Baseline (`nbody_serial`)
```bash
cd backends/serial
g++ -O3 -std=c++17 -I../../common/cpp nbody_serial.cpp -o nbody_serial
chmod +x nbody_serial
cd ../..
```

---

## 4. Generate Initial Conditions (100 to 50,000 Bodies)

Generate reproducible Solar-system IC datasets into `data/ic/`:
```bash
python3 common/python/export_ic.py --out-dir data/ic
```
*(Creates `ic_N100_s42.csv` through `ic_N50000_s42.csv`).*

---

## 5. Quick Verification Run & Accuracy Check

### Step 1: Run 100 Steps on $N=1,000$ Bodies
```bash
mkdir -p backends/serial/out backends/cuda/out bench/results/cpp-serial/local bench/results/cuda/ubuntu-test

# 1. Run CPU Serial Baseline
./backends/serial/nbody_serial \
  --input data/ic/ic_N1000_s42.csv \
  --steps 100 \
  --final-state backends/serial/out/serial_1000.csv \
  --output-json bench/results/cpp-serial/local/serial_1000.json

# 2. Run CUDA GPU (Tiled Double-Precision FP64)
./backends/cuda/nbody_cuda \
  --input data/ic/ic_N1000_s42.csv \
  --steps 100 \
  --variant tiled-f64 \
  --final-state backends/cuda/out/cuda_1000.csv \
  --output-json bench/results/cuda/ubuntu-test/cuda_1000.json \
  --machine ubuntu-test
```

### Step 2: Validate Numerical Accuracy
```bash
python3 bench/compare.py \
  --serial-json bench/results/cpp-serial/local/serial_1000.json \
  --gpu-json bench/results/cuda/ubuntu-test/cuda_1000.json \
  --serial-csv backends/serial/out/serial_1000.csv \
  --gpu-csv backends/cuda/out/cuda_1000.csv \
  --tol 1e-9
```
Expected output:
```text
[VALIDATION] Numerical Accuracy: PASS
  Max Position Absolute Difference: ~4.44e-16 AU (Tolerance: 1.00e-09 AU)
```

---

## 6. Run the Full Automated Benchmark Sweep

Run the full parameter sweep (sweeps across all $N \in \{100, 500, 1000, 2000, 5000, 10000, 20000, 50000\}$, block sizes, FP64, FP32, and host-device transfer costs):

```bash
python3 backends/cuda/sweep_cuda.py \
  --machine ubuntu-$(hostname) \
  --steps 100 \
  --repeats 3
```

### What this Command Does Automatically:
1. Detects and executes the compiled `nbody_cuda` and `nbody_serial` binaries.
2. Benchmarks the CPU scalar baseline for each body count $N$.
3. Runs 38 benchmark configurations $\times$ 3 repeats on the GPU.
4. Saves all raw timing telemetry as JSON files into:
   ```
   bench/results/cuda/ubuntu-<hostname>/
   ```
5. **Automatically calculates GPU speed-ups and writes the Markdown report:**
   ```
   bench/results/cuda/ubuntu-<hostname>/BENCHMARK_REPORT.md
   ```

---

## 7. All-in-One Ubuntu Benchmark Script

You can run this self-contained bash script directly to execute the entire pipeline from scratch:

```bash
#!/usr/bin/env bash
set -e

echo "=== 1. Activating Python Environment ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q numpy pandas

echo "=== 2. Compiling C++ Serial Baseline ==="
cd backends/serial
g++ -O3 -std=c++17 -I../../common/cpp nbody_serial.cpp -o nbody_serial
chmod +x nbody_serial
cd ../..

echo "=== 3. Compiling CUDA GPU Binary ==="
# Detect GPU architecture or default to sm_75
ARCH="sm_75"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    echo "Detected GPU: $GPU_NAME"
    if [[ "$GPU_NAME" == *"1080"* || "$GPU_NAME" == *"P100"* ]]; then ARCH="sm_60"; fi
    if [[ "$GPU_NAME" == *"V100"* ]]; then ARCH="sm_70"; fi
    if [[ "$GPU_NAME" == *"30"* || "$GPU_NAME" == *"A10"* ]]; then ARCH="sm_86"; fi
    if [[ "$GPU_NAME" == *"40"* ]]; then ARCH="sm_89"; fi
fi
echo "Using NVCC Arch: -$ARCH"

cd backends/cuda
nvcc -O3 -std=c++17 -arch=$ARCH -I../../common/cpp nbody_cuda.cu -o nbody_cuda
chmod +x nbody_cuda
cd ../..

echo "=== 4. Ensuring Initial Conditions Exist ==="
python3 common/python/export_ic.py --out-dir data/ic

echo "=== 5. Running Automated Benchmark Sweep ==="
MACHINE_ID="ubuntu-$(hostname)"
python3 backends/cuda/sweep_cuda.py --machine "$MACHINE_ID" --steps 100 --repeats 3

echo "=== Benchmark Complete! ==="
echo "Report generated at: bench/results/cuda/$MACHINE_ID/BENCHMARK_REPORT.md"
```
