# Unified Architecture Contract and Disparate Hardware Benchmark Integration

## Status
Accepted

## Context
Three project branches independently developed distinct parallel backends for the N-body simulation: OpenMP on an Intel Core i5-1340P (`origin/siam`), MPI with `mpi4py` on an Apple M4 (`origin/mnb`), and CUDA on an Ubuntu host with an NVIDIA GeForce RTX 5090 (`origin/shadhin_dev`). Integrating these backends into a unified codebase presented discrepancies in result serialization schemas, scalar baseline organization, and inability to run CUDA or multi-core MPI benchmarks on the integration lead's AMD Ryzen host.

## Decision
1. Standardize on **Contract v1** (`common/CONTRACT.md`) for all shared C++ and Python I/O, maintaining the canonical flat schema (`schema_version: 1`, `n_bodies`, `t_loop_s`, etc.) required by `bench/check_result.py`.
2. Establish a dedicated `backends/serial/` module containing `nbody_serial.cpp` as the clean standalone C++ scalar baseline, while preserving `backends/openmp/` for multi-threaded CPU variants.
3. Ingest and preserve the native hardware benchmark runs committed by each team member under `bench/results/<backend>/<machine>/` rather than attempting local re-benchmarks. Speedup and efficiency curves are evaluated against the respective machine's anchor baseline.
4. Extract Mansib's 780 MPI benchmark JSONs from `py-mpi.zip` into `bench/results/py-mpi/mansib-m4/` and generate the comprehensive `BENCHMARK_REPORT.md` using `summarize_mpi.py`.

## Consequences
- Every parallel backend (Serial, OpenMP, MPI, CUDA) is callable with a standardized CLI interface and produces verified, comparable timing artifacts.
- The project retains rich architectural diversity across modern microarchitectures (Intel Raptor Lake hybrid P/E-cores, Apple M4 ARM, NVIDIA Blackwell/Ada RTX 5090) without requiring identical hardware across all team machines.
- All implementations satisfy the M1 correctness gate with relative error $< 10^{-10}$ against golden reference trajectories.
