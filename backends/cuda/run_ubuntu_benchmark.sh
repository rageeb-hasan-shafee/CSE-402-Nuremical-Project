#!/usr/bin/env bash
# backends/cuda/run_ubuntu_benchmark.sh
# Automated compile, execution, and benchmark sweep script for Ubuntu Linux.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================================="
echo "   CSE-402 CUDA N-Body Simulation — Ubuntu Benchmark      "
echo "=========================================================="

# 1. Python environment
if [ ! -d ".venv" ]; then
    echo "[1/5] Creating Python virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q numpy pandas

# 2. Compile C++ Serial Baseline
echo "[2/5] Compiling single-core C++ scalar baseline..."
cd "$REPO_ROOT/backends/serial"
g++ -O3 -std=c++17 -I../../common/cpp nbody_serial.cpp -o nbody_serial
chmod +x nbody_serial

# 3. Detect GPU & Compile CUDA
echo "[3/5] Detecting GPU architecture and compiling CUDA..."
ARCH="sm_75"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    echo "       Detected GPU: $GPU_NAME"
    if [[ "$GPU_NAME" == *"1080"* || "$GPU_NAME" == *"P100"* ]]; then ARCH="sm_60"; fi
    if [[ "$GPU_NAME" == *"V100"* ]]; then ARCH="sm_70"; fi
    if [[ "$GPU_NAME" == *"30"* || "$GPU_NAME" == *"A10"* ]]; then ARCH="sm_86"; fi
    if [[ "$GPU_NAME" == *"40"* ]]; then ARCH="sm_89"; fi
fi
echo "       Using NVCC Flag: -arch=$ARCH"

cd "$REPO_ROOT/backends/cuda"
nvcc -O3 -std=c++17 -arch=$ARCH -I../../common/cpp nbody_cuda.cu -o nbody_cuda
chmod +x nbody_cuda
cd "$REPO_ROOT"

# 4. Generate Initial Conditions if not already present
echo "[4/5] Checking initial conditions dataset..."
python3 common/python/export_ic.py --out-dir data/ic

# 5. Run Benchmark Sweep
MACHINE_ID="ubuntu-$(hostname)"
echo "[5/5] Running automated benchmark sweep (Machine ID: $MACHINE_ID)..."
python3 backends/cuda/sweep_cuda.py --machine "$MACHINE_ID" --steps 100 --repeats 3

echo ""
echo "=========================================================="
echo " Benchmark Complete!"
echo " Results and JSONs saved to: bench/results/cuda/$MACHINE_ID/"
echo " Markdown Report generated at: bench/results/cuda/$MACHINE_ID/BENCHMARK_REPORT.md"
echo "=========================================================="
