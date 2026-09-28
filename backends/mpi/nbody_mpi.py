"""MPI N-body backend (py-mpi). Contract v1. Owner: Mansib.

Distributed-memory velocity Verlet after Zhu (2020): every rank owns a
contiguous block of bodies, computes the forces on that block from all N
bodies, and the ranks exchange positions once per step.

    mpiexec -n 4 python backends/mpi/nbody_mpi.py --input data/ic/ic_N1000_s42.csv \
        --steps 100 --variant allgather --machine mansib-m4 --output-json out.json

Variants:
    allgather      one Allgatherv per step (default)
    master-worker  Gatherv to rank 0, then Bcast from rank 0 (the paper's design)
"""

import os
import pathlib
import sys

# One thread per rank. This must happen BEFORE numpy is imported, otherwise a
# multithreaded BLAS inside every rank oversubscribes the cores and ruins the scaling.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

if sys.platform == "win32":
    for _p in (
        r"E:\system\scoop\apps\msmpi\10.1.1",
        r"C:\Program Files\Microsoft MPI\Bin",
        r"C:\Program Files\Microsoft MPI",
    ):
        if os.path.exists(_p):
            os.add_dll_directory(_p)

import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import contract_io as nbio  # noqa: E402

VARIANTS = ("allgather", "master-worker")


def decompose(n, size):
    """Block decomposition: the first n % size ranks get one extra body.

    Returns (counts, displs): rank r owns bodies displs[r] .. displs[r] + counts[r] - 1.
    """
    counts = np.array([n // size + (1 if r < n % size else 0) for r in range(size)])
    displs = np.concatenate(([0], np.cumsum(counts)[:-1]))
    return counts, displs


def accel_rows(pos, m, lo, hi, chunk=256):
    """Accelerations on bodies lo..hi-1 from ALL N bodies, vectorised in row chunks.

    Chunking keeps the temporary (chunk, N, 3) array small: 256 x 10000 x 3 doubles = 61 MB.
    """
    out = np.empty((hi - lo, 3))
    for a in range(lo, hi, chunk):
        b = min(a + chunk, hi)
        diff = pos[None, :, :] - pos[a:b, None, :]  # (c, N, 3): r_j - r_p
        d2 = np.einsum("ijk,ijk->ij", diff, diff)  # squared distances
        rows = np.arange(b - a)
        d2[rows, a + rows] = np.inf  # self-term: inf**-1.5 == 0
        out[a - lo : b - lo] = nbio.G * np.einsum(
            "ij,ijk->ik", d2**-1.5 * m[None, :], diff
        )
    return out


def exchange_positions(comm, variant, pos, lo, hi, c3, d3):
    """After this call every rank holds the up-to-date positions of ALL bodies."""
    if variant == "allgather":
        comm.Allgatherv(MPI.IN_PLACE, [pos, c3, d3, MPI.DOUBLE])
    else:  # master-worker, as described in Zhu (2020) §3
        if comm.Get_rank() == 0:
            comm.Gatherv(MPI.IN_PLACE, [pos, c3, d3, MPI.DOUBLE], root=0)
        else:
            comm.Gatherv(pos[lo:hi], None, root=0)
        comm.Bcast(pos, root=0)


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    args = nbio.parse_args(default_variant="allgather")
    if args.variant not in VARIANTS:
        if rank == 0:
            print(
                f"unknown --variant {args.variant!r}; expected one of {VARIANTS}",
                file=sys.stderr,
            )
        sys.exit(2)  # every rank sees the same args, so all exit

    # Rank 0 reads the file (not timed, contract §C5); a failure must stop every rank.
    names = data = error = None
    if rank == 0:
        try:
            names, *data = nbio.read_ic_csv(args.input)
        except Exception as exc:  # noqa: BLE001 - report and stop all ranks
            error = f"{type(exc).__name__}: {exc}"
    error = comm.bcast(error, root=0)
    if error:
        if rank == 0:
            print(f"cannot read --input: {error}", file=sys.stderr)
        sys.exit(1)

    # ---- setup (timed): broadcast, decompose, initial acceleration a0 ----
    comm.Barrier()
    t0 = MPI.Wtime()
    m, pos, vel = comm.bcast(data, root=0)
    n = len(m)
    if n < size:
        if rank == 0:
            print(f"need at least one body per rank (N={n}, P={size})", file=sys.stderr)
        sys.exit(2)
    counts, displs = decompose(n, size)
    lo, hi = int(displs[rank]), int(displs[rank] + counts[rank])
    c3, d3 = counts * 3, displs * 3  # MPI counts in doubles: 3 per body
    acc = accel_rows(pos, m, lo, hi)
    t_setup = MPI.Wtime() - t0

    # ---- timed integration loop ----
    dt = args.dt
    t_comm = t_force = 0.0
    comm.Barrier()
    t0 = MPI.Wtime()
    for _ in range(args.steps):
        pos[lo:hi] += vel[lo:hi] * dt + 0.5 * acc * dt * dt  # drift: own slice only

        tc = MPI.Wtime()
        exchange_positions(
            comm, args.variant, pos, lo, hi, c3, d3
        )  # the only communication
        t_comm += MPI.Wtime() - tc

        tf = MPI.Wtime()
        new = accel_rows(pos, m, lo, hi)  # O(N^2 / P) work
        t_force += MPI.Wtime() - tf

        vel[lo:hi] += 0.5 * (acc + new) * dt  # kick: own slice only
        acc = new
    comm.Barrier()
    t_loop = MPI.Wtime() - t0

    # Velocities are only up to date on each rank's own slice: collect them on rank 0.
    # (Positions are already complete everywhere after the last exchange.)
    if rank == 0:
        comm.Gatherv(MPI.IN_PLACE, [vel, c3, d3, MPI.DOUBLE], root=0)
    else:
        comm.Gatherv(vel[lo:hi], None, root=0)

    # Contract §C5: report the slowest rank.
    t_setup = comm.reduce(t_setup, op=MPI.MAX, root=0)
    t_loop_max = comm.reduce(t_loop, op=MPI.MAX, root=0)
    t_force_max = comm.reduce(t_force, op=MPI.MAX, root=0)
    t_comm_max = comm.reduce(t_comm, op=MPI.MAX, root=0)
    t_comm_sum = comm.reduce(t_comm, op=MPI.SUM, root=0)
    hosts = comm.gather(MPI.Get_processor_name(), root=0)

    if rank == 0:
        if args.final_state:
            nbio.write_state_csv(args.final_state, names, m, pos, vel)
        nbio.write_result_json(
            args,
            "py-mpi",
            "python",
            n,
            size,
            t_setup,
            t_loop_max,
            t_force=t_force_max,
            t_comm=t_comm_max,
            extra={
                "t_comm_mean_s": t_comm_sum / size,
                "bodies_per_rank_min": int(counts.min()),
                "bodies_per_rank_max": int(counts.max()),
                "mpi_vendor": "%s %s"
                % (MPI.get_vendor()[0], ".".join(map(str, MPI.get_vendor()[1]))),
                "processor_names": sorted(set(hosts)),
                "n_hosts": len(set(hosts)),
            },
        )
        share = 100 * t_comm_max / t_loop_max if t_loop_max > 0 else 0.0
        print(
            f"py-mpi[{args.variant}] N={n} P={size} steps={args.steps} loop={t_loop_max:.4f}s "
            f"({t_loop_max / max(args.steps, 1) * 1e3:.3f} ms/step, comm {share:.1f}%)"
        )


if __name__ == "__main__":
    main()
