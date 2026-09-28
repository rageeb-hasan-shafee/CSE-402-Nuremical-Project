"""Benchmark sweep for the MPI backend: one JSON per run, resumable. Run from the repo root.

    python backends/mpi/sweep_local.py --machine mansib-m4                 # full local matrix
    python backends/mpi/sweep_local.py --machine mansib-m4 --quick         # small smoke sweep
    python backends/mpi/sweep_local.py --machine mansib-lab-x2 \
        --hostfile backends/mpi/hostfile --procs 1,2,4,8,16 --mpi-args "--mca btl_tcp_if_include eth0"

Output: bench/results/py-mpi/<machine>/py-mpi_<variant>_N<N>_P<P>_r<k>.json
r0 is the warm-up run (contract §C5); the summary script ignores it. Existing
files are skipped, so an interrupted sweep continues where it stopped.
"""
import argparse
import pathlib
import shlex
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from test_data import find_or_make_ic  # noqa: E402

FULL = dict(ns="100,500,1000,2000,5000,10000", procs="1,2,4,6,8,10", variants="allgather,master-worker")
QUICK = dict(ns="100,1000", procs="1,2,4", variants="allgather")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", required=True, help="tag starting with your name, e.g. mansib-m4")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ns")
    ap.add_argument("--procs")
    ap.add_argument("--variants")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=5, help="timed repeats (N >= 5000 uses min(3, this))")
    ap.add_argument("--hostfile", help="for cluster runs")
    ap.add_argument("--mpi-args", default="", help="extra mpiexec arguments, quoted")
    ap.add_argument("--oversubscribe", action="store_true", help="allow more ranks than cores")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    preset = QUICK if a.quick else FULL
    ns = [int(x) for x in (a.ns or preset["ns"]).split(",")]
    procs = [int(x) for x in (a.procs or preset["procs"]).split(",")]
    variants = (a.variants or preset["variants"]).split(",")
    if not a.machine.startswith("mansib-"):
        print(f"warning: machine tag {a.machine!r} does not start with 'mansib-' (ownership rule, ULTRA §4)")

    out_dir = ROOT / "bench" / "results" / "py-mpi" / a.machine
    if not a.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    launcher = ["mpiexec"]
    if a.hostfile:
        launcher += ["--hostfile", str(pathlib.Path(a.hostfile).resolve())]
    if a.oversubscribe:
        launcher += ["--oversubscribe"]
    launcher += shlex.split(a.mpi_args)

    jobs = []
    for n in ns:
        ic, kind = find_or_make_ic(n)
        if kind == "test":
            print(f"note: N={n} uses a TEST IC ({ic.relative_to(ROOT)}); rerun with data/ic files for final numbers")
        reps = min(3, a.repeats) if n >= 5000 else a.repeats
        for variant in variants:
            for p in procs:
                for r in range(reps + 1):                          # r0 = warm-up
                    out = out_dir / f"py-mpi_{variant}_N{n}_P{p}_r{r}.json"
                    jobs.append((n, p, variant, r, ic, out))

    todo = [j for j in jobs if not j[5].exists()]
    print(f"{len(jobs)} runs in matrix, {len(jobs) - len(todo)} already done, {len(todo)} to run → {out_dir.relative_to(ROOT)}")
    t_start = time.time()
    for i, (n, p, variant, r, ic, out) in enumerate(todo, 1):
        cmd = launcher + ["-n", str(p), sys.executable, str(HERE / "nbody_mpi.py"),
                          "--input", str(ic), "--steps", str(a.steps), "--variant", variant,
                          "--machine", a.machine, "--output-json", str(out)]
        label = f"[{i}/{len(todo)}] N={n} P={p} {variant} r{r}"
        if a.dry_run:
            print(label, " ".join(shlex.quote(c) for c in cmd))
            continue
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
        if res.returncode != 0:
            out.unlink(missing_ok=True)
            print(f"{label} FAILED\n{res.stderr.strip()[-500:]}")
            continue
        print(f"{label}  {res.stdout.strip()}")
    if not a.dry_run:
        print(f"done in {time.time() - t_start:.0f}s. Summarise with:\n"
              f"  python backends/mpi/summarize_mpi.py --machine {a.machine}")


if __name__ == "__main__":
    main()
