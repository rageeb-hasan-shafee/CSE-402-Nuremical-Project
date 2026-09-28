# Dual-Anchoring Performance Normalization and Supervisor Defense Demonstration Platform

## Status
Accepted

## Context
The project benchmarked gravitational N-body simulations across heterogeneous platforms and architectures:
1. **Intel Core i5-1340P** (4P + 8E cores, 16 threads): C++ OpenMP variants (`static`, `dynamic`, `simd`, `newton3`).
2. **Apple M4** (10-core ARM): Python MPI (`mpi4py`) collective (`allgather`) and point-to-point (`ring`, `master-worker`).
3. **NVIDIA GeForce RTX 5090** (Blackwell/Ada): CUDA GPU kernels (`naive`, `tiled-f64`, `tiled-f32`, `unroll-tiled`).
4. **AMD Ryzen 5 5600G** (6C/12T Zen 3): C++ standalone serial, OpenMP, and MS-MPI running on Windows.

Evaluating parallel speedup solely relative to each machine's native single-thread baseline ($S_{native} = T_{native}(1) / T_{native}(P)$) is appropriate for intra-architecture scaling (e.g. Amdahl's Law, memory bandwidth saturation), but obscures cross-platform throughput comparisons (e.g. evaluating an RTX 5090 or Apple M4 w.r.t standard desktop CPU performance). Furthermore, the Week 14 project evaluation mandates an individual supervisor oral defense and hands-on live code/simulation demonstration on the student's laptop without setup delays.

## Decision
1. **Dual-Anchoring Methodology**:
   - **Native Host Anchor**: Normalized against the respective hardware's single-core execution time ($T_{host}(P=1)$). Captures speedup factor, parallel efficiency ($\eta = S/P$), and thread/process scalability.
   - **Universal Hardware Anchor**: Grounded against Fahad's AMD Ryzen 5 5600G scalar baseline ($T_{Ryzen}(P=1)$). Normalizes absolute throughput across architectures, demonstrating how GPU acceleration (e.g., RTX 5090 tiled-f32 achieving up to $>330\times$ speedup) and multi-node/multi-process models compare against desktop CPU performance.
2. **Consolidated Data Ingestion**:
   - Process all JSON benchmark results across all 4 platforms and export a consolidated dataset (`bench/results/consolidated_benchmarks.json`) alongside publication-grade Matplotlib plots in `bench/plots/`.
3. **Interactive Supervisor Oral Defense & Demonstration Dashboard**:
   - Provide a standalone, zero-dependency browser application (`bench/defense_dashboard.html`) designed specifically for the Week 14 oral defense requirements:
     - **Slide Policy Visual Aids (Slides 1–6)**: Base paper formulation (Tailin Zhu 2020), 1PN General Relativity precession, unified architecture, and individual contributions.
     - **Live Simulation Demo**: Real-time HTML5 2D Canvas N-body integrator with customizable asteroid belt density, planetary orbits, and continuous Hamiltonian energy conservation readout ($\Delta E / E_0 < 10^{-7}$).
     - **Comparative Benchmark Explorer**: Dynamic Chart.js interactive graphs supporting instant switching between Native Anchor and Universal Ryzen Anchor, runtime vs speedup metrics, and log-log scaling curves.
     - **Supervisor Q&A Cheat Sheet**: Pre-formulated defenses for core HPC and numerical analysis questions (symplectic Verlet vs RK4, OpenMP thread barrier latency at $N=100$, Intel P/E hybrid vs AMD Zen 3 scheduling, CUDA shared memory tiling vs FP64 hardware limits, and MPI collective communication overhead).

## Consequences
- The group possesses a unified, authoritative evaluation platform suitable for both formal documentation and live demonstration.
- Cross-hardware comparisons remain methodologically transparent without misleading or mismatched speedup claims.
- The oral defense requirements (pre-configured local execution, live code/simulation demo, individual contribution walkthrough, and technical Q&A defense) are fully supported.
