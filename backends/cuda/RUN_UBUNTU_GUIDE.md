# CUDA N-Body Simulation — Ubuntu Benchmark Guide (RTX 5090 / Custom Path)

This guide is tailored for running and benchmarking the CUDA GPU and C++ Serial N-Body simulation on **Ubuntu 24.04 LTS** with the **NVIDIA GeForce RTX 5090** (Blackwell Architecture, `sm_120`), using the isolated CUDA Toolkit installed in `/mnt/models/script_checking/cuda-toolkit`.

---

## 0. Current Hardware & Environment Profile

| Component | Value on this Machine |
|:---|:---|
| **OS** | Ubuntu 24.04.4 LTS (Noble Numbat) |
| **GPU** | NVIDIA GeForce RTX 5090 (32 GB VRAM) |
| **Driver Version** | 580.159.03 |
| **CUDA Compute Arch** | **`sm_120`** (Blackwell) |
| **CUDA Toolkit Path** | `/mnt/models/script_checking/cuda-toolkit` |
| **Project Workspace** | `/mnt/models/script_checking/CSE-402-Nuremical-Project` |

---

## 1. Prerequisites & Environment Setup

### Step 1: Verify NVIDIA Driver & GPU
Ensure your RTX 5090 is active:
```bash
nvidia-smi
```
*(You should see `NVIDIA GeForce RTX 5090`, Driver `580.159.03`, and CUDA `13.0` support).*

### Step 2: Ensure CUDA 12.8 Toolkit is in PATH
Since CUDA is installed in `/mnt/models/script_checking/cuda-toolkit` (to preserve disk space on root `/`), ensure your shell session has it exported:

```bash
# Export to current shell session:
export PATH=/mnt/models/script_checking/cuda-toolkit/bin:$PATH
export LD_LIBRARY_PATH=/mnt/models/script_checking/cuda-toolkit/lib64:$LD_LIBRARY_PATH

# (Optional) Persist permanently in ~/.bashrc:
if ! grep -q "cuda-toolkit" ~/.bashrc; then
  echo 'export PATH=/mnt/models/script_checking/cuda-toolkit/bin:$PATH' >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=/mnt/models/script_checking/cuda-toolkit/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
fi
```

Verify `nvcc` and `g++`:
```bash
g++ --version
nvcc --version
```
*(Expected: `nvcc` release 12.8, V12.8.x and `g++` 13.x).*

### Step 3: Activate Python Virtual Environment
From the repository root (`/mnt/models/script_checking/CSE-402-Nuremical-Project`):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas matplotlib
```

---

## 2. GPU Architecture Flag (`-arch=sm_120`)

Your **RTX 5090** is built on NVIDIA's **Blackwell** architecture.  
Always use **`-arch=sm_120`** (or `-arch=native`) when invoking `nvcc`.

| GPU Architecture | Models | NVCC Architecture Flag |
|:---|:---|:---:|
| **Blackwell (This Device)** | **RTX 5090 / 5080 / 5070** | **`-arch=sm_120`** *(or `-arch=native`)* |
| **Ada Lovelace** | RTX 4090 / 4080 / L4 | `-arch=sm_89` |
| **Ampere** | RTX 3090 / 3080 / A100 | `-arch=sm_86` / `-arch=sm_80` |
| **Turing** | RTX 2080 / GTX 1660 / T4 | `-arch=sm_75` |
| **Pascal** | GTX 1080 / Tesla P100 | `-arch=sm_60` |

---

## 3. Compile Both Binaries

### 1. Compile CUDA GPU Binary for RTX 5090 (`nbody_cuda`)
```bash
cd backends/cuda
nvcc -O3 -std=c++17 -arch=sm_120 -I../../common/cpp nbody_cuda.cu -o nbody_cuda
chmod +x nbody_cuda
cd ../..
```

### 2. Compile Single-Core C++ Serial Baseline (`nbody_serial`)
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
*(This generates `ic_N100_s42.csv` through `ic_N50000_s42.csv`).*

---

## 5. Quick Verification Run & Accuracy Check

### Step 1: Run 100 Steps on $N=1,000$ Bodies
```bash
mkdir -p backends/serial/out backends/cuda/out bench/results/cpp-serial/local bench/results/cuda/rtx5090-test

# 1. Run CPU Serial Baseline
./backends/serial/nbody_serial \
  --input data/ic/ic_N1000_s42.csv \
  --steps 100 \
  --final-state backends/serial/out/serial_1000.csv \
  --output-json bench/results/cpp-serial/local/serial_1000.json

# 2. Run CUDA GPU on RTX 5090 (Tiled Double-Precision FP64)
./backends/cuda/nbody_cuda \
  --input data/ic/ic_N1000_s42.csv \
  --steps 100 \
  --variant tiled-f64 \
  --final-state backends/cuda/out/cuda_1000.csv \
  --output-json bench/results/cuda/rtx5090-test/cuda_1000.json \
  --machine rtx5090-test
```

### Step 2: Validate Numerical Accuracy (GPU vs Serial CPU)
```bash
python3 bench/compare.py \
  --serial-json bench/results/cpp-serial/local/serial_1000.json \
  --gpu-json bench/results/cuda/rtx5090-test/cuda_1000.json \
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
  --machine ubuntu-rtx5090 \
  --steps 100 \
  --repeats 3
```

### What this Command Does Automatically:
1. Detects and executes the compiled `nbody_cuda` and `nbody_serial` binaries.
2. Benchmarks the CPU scalar baseline for each body count $N$.
3. Runs 38 benchmark configurations $\times$ 3 repeats on the RTX 5090.
4. Saves all raw timing telemetry as JSON files into:
   ```
   bench/results/cuda/ubuntu-rtx5090/
   ```
5. **Automatically calculates GPU speed-ups and writes the Markdown report:**
   ```
   bench/results/cuda/ubuntu-rtx5090/BENCHMARK_REPORT.md
   ```

---

## 7. All-in-One Ubuntu Benchmark Script

You can execute the entire pipeline with a single command using the included script:

```bash
./backends/cuda/run_ubuntu_benchmark.sh
```

Or run this self-contained bash script:

```bash
#!/usr/bin/env bash
set -e

# Ensure isolated CUDA toolkit is accessible in PATH
if [ -d "/mnt/models/script_checking/cuda-toolkit/bin" ]; then
    export PATH=/mnt/models/script_checking/cuda-toolkit/bin:$PATH
    export LD_LIBRARY_PATH=/mnt/models/script_checking/cuda-toolkit/lib64:$LD_LIBRARY_PATH
fi

echo "=== 1. Activating Python Environment ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q numpy pandas matplotlib

echo "=== 2. Compiling C++ Serial Baseline ==="
cd backends/serial
g++ -O3 -std=c++17 -I../../common/cpp nbody_serial.cpp -o nbody_serial
chmod +x nbody_serial
cd ../..

echo "=== 3. Compiling CUDA GPU Binary for RTX 5090 ==="
ARCH="sm_120"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    echo "Detected GPU: $GPU_NAME"
    if [[ "$GPU_NAME" == *"1080"* || "$GPU_NAME" == *"P100"* ]]; then ARCH="sm_60"; fi
    if [[ "$GPU_NAME" == *"V100"* ]]; then ARCH="sm_70"; fi
    if [[ "$GPU_NAME" == *"30"* || "$GPU_NAME" == *"A10"* ]]; then ARCH="sm_86"; fi
    if [[ "$GPU_NAME" == *"40"* ]]; then ARCH="sm_89"; fi
    if [[ "$GPU_NAME" == *"50"* ]]; then ARCH="sm_120"; fi
fi
echo "Using NVCC Arch: -$ARCH"

cd backends/cuda
nvcc -O3 -std=c++17 -arch=$ARCH -I../../common/cpp nbody_cuda.cu -o nbody_cuda
chmod +x nbody_cuda
cd ../..

echo "=== 4. Ensuring Initial Conditions Exist ==="
python3 common/python/export_ic.py --out-dir data/ic

echo "=== 5. Running Automated Benchmark Sweep ==="
MACHINE_ID="ubuntu-rtx5090"
python3 backends/cuda/sweep_cuda.py --machine "$MACHINE_ID" --steps 100 --repeats 3

echo "=== Benchmark Complete! ==="
echo "Report generated at: bench/results/cuda/$MACHINE_ID/BENCHMARK_REPORT.md"
```

---

## 8. Completely Removing CUDA Later

When you are finished with your project and want to recover all disk space:

```bash
# 1. Delete the CUDA installation directory
rm -rf /mnt/models/script_checking/cuda-toolkit

# 2. Delete the installer download & cache (if still present)
rm -rf /mnt/models/script_checking/cuda_installer

# 3. Clean up PATH in ~/.bashrc
sed -i '/cuda-toolkit/d' ~/.bashrc
source ~/.bashrc
```
*(See also [HOW_TO_REMOVE_CUDA.txt](../../HOW_TO_REMOVE_CUDA.txt)).*
