# MPI backend (`py-mpi`) — owner: Mansib

Distributed-memory N-body simulation (velocity Verlet) with **mpi4py**, after
Zhu (2020). Each MPI rank owns a block of bodies, computes the forces on that
block from all bodies, and ranks exchange positions once per step.

| File | Purpose |
|---|---|
| `nbody_mpi.py` | the backend. Variants: `allgather` (default), `master-worker` (the paper's design) |
| `contract_io.py` | contract I/O (CLI flags, IC/final-state CSV, result JSON). Uses `common/python/nbio.py` automatically once that exists |
| `test_data.py` | test ICs + reference answers from the original `nbody_sim` serial integrator |
| `verify_mpi.py` | step-by-step correctness check (run this first) |
| `sweep_local.py` | benchmark sweep → `bench/results/py-mpi/<machine>/` (resumable) |
| `summarize_mpi.py` | speed-up / efficiency / comm-share / Amdahl table, optional plots |
| `hostfile.example` | template for cluster runs |

## 1. Install (once per machine)

System MPI first, then mpi4py inside the project venv (`nbody_sim/.venv`):

| OS | System MPI |
|---|---|
| macOS | `brew install open-mpi` |
| Ubuntu/Debian | `sudo apt install openmpi-bin libopenmpi-dev` |
| Windows | Microsoft MPI: install **both** `msmpisetup.exe` and `msmpisdk.msi`, then open a new terminal |

```bash
# from the repo root; Windows: nbody_sim\.venv\Scripts\python
nbody_sim/.venv/bin/python -m pip install -r backends/mpi/requirements-mpi.txt
```

## 2. Verify (must print ALL CHECKS PASSED)

```bash
nbody_sim/.venv/bin/python backends/mpi/verify_mpi.py
```

## 3. Run once

```bash
mpiexec -n 4 nbody_sim/.venv/bin/python backends/mpi/nbody_mpi.py \
    --input data/ic/ic_N1000_s42.csv --steps 100 --variant allgather \
    --machine mansib-m4 --output-json out.json --final-state final.csv
```
If `data/ic/` isn't on your branch yet, use a test IC:
`backends/mpi/out/ic/test_ic_N1000_s42.csv` (created by `verify_mpi.py`).

## 4. Benchmark sweep + summary

```bash
nbody_sim/.venv/bin/python backends/mpi/sweep_local.py --machine mansib-m4          # ~1-1.5 h on an M4 (N=10000 dominates)
nbody_sim/.venv/bin/python backends/mpi/summarize_mpi.py --machine mansib-m4 --report   # → bench/results/py-mpi/mansib-m4/BENCHMARK_REPORT.md
```
The root `.gitignore` currently ignores every `results/` folder. Until Fahad
changes that (M0), commit result JSONs with
`git add -f bench/results/py-mpi/mansib-m4/`.

## 5. Cluster (2+ Linux machines)

Same Open MPI version, Python 3.11 and repo path on every node; passwordless
SSH from the head node. Then:
```bash
cp backends/mpi/hostfile.example backends/mpi/hostfile      # edit hostnames / slots
mpiexec --hostfile backends/mpi/hostfile -n 16 hostname     # every node must appear
nbody_sim/.venv/bin/python backends/mpi/sweep_local.py --machine mansib-lab-x2 \
    --hostfile backends/mpi/hostfile --procs 1,2,4,8,16 --ns 5000,10000 \
    --mpi-args "--mca btl_tcp_if_include eth0"
```
The `processor_names` / `n_hosts` fields in each result JSON show how many
machines a run really used.

## Machines (hardware for each `--machine` tag)

| Tag | Hardware |
|---|---|
| `mansib-m4` | Apple M4 MacBook Air, 10 cores (4 performance + 6 efficiency), 16 GB, macOS 26, Open MPI 5.0.11, Python 3.11, mpi4py 4.1 |
