# CONTRACT (v1)

Contract version: 1
Project: GPU-Accelerated N-body Simulation of the Solar System (CSE 402, Group C_G8)

All four backends (py-numpy / py-loop, cpp-serial, cpp-openmp, py-mpi, cuda) must strictly follow these rules.

## C1. Constants and Units
- Length: AU
- Time: day
- Mass: Solar Mass (Msun)
- G = 2.9591220828559115e-4 (= 0.01720209895^2, Gaussian constant squared)
- Softening: 0.0 (self-interaction skipped by index j == p)
- Default dt: 0.1 day
- Default steps: 100
- Precision: float64 for all backends (CUDA float32 is an extra experiment)

## C2. Initial Conditions
- Built by `common/python/export_ic.py` into `data/ic/ic_N{N}_s42.csv` (seed=42).
- Bodies 0..9: Sun, 8 planets, Moon from `nbody_sim/orbital_elements.py` at J2000.
- Bodies 10..N-1: synthetic main-belt asteroids (mass 1e12 to 3e17 kg).
- System shifted to barycentric frame (zero total momentum).
- CSV Header: `name,mass,x,y,z,vx,vy,vz`
- Floating-point numbers written with full precision (`%.17g` in C++, `repr()` in Python).

## C3. Command-Line Interface (every executable or script)
```
<program> --input data/ic/ic_N1000_s42.csv
          --steps 100 --dt 0.1
          --workers P            # OpenMP threads | ignored by MPI | 1 for CUDA
          --variant NAME         # static, dynamic, simd, newton3, etc.
          --block-size B         # CUDA only (default 256)
          --machine TAG          # e.g. siam-i5-1340p, mansib-m4, shadhin-colab-t4
          --output-json PATH     # result JSON (§C4)
          --final-state PATH     # final-state CSV (§C2 format)
```

## C4. Result JSON (schema v1)
```json
{
  "schema_version": 1,
  "backend": "cpp-openmp",
  "variant": "static",
  "lang": "cpp",
  "n_bodies": 1000,
  "workers": 8,
  "steps": 100,
  "dt": 0.1,
  "t_setup_s": 0.0123,
  "t_loop_s": 1.2345,
  "t_force_s": 1.2001,
  "t_comm_s": 0.0,
  "t_transfer_s": 0.0,
  "machine": "siam-i5-1340p",
  "extra": {}
}
```

## C5. Timing Rules
- `t_setup_s`: memory allocation, initial a0 computation. Excludes reading CSV.
- `t_loop_s`: integration loop only (primary metric).
- Timers: monotonic wall-clock (`std::chrono::steady_clock`, `time.perf_counter()`).
- Repeat rule: 1 warmup run + 5 timed runs (median reported).

## C6. Correctness Gate
Max relative position and velocity error < 1e-10 against reference data:
```
python bench/validate.py --test <final.csv> --ref data/ref/final_N100_steps10.csv
```
