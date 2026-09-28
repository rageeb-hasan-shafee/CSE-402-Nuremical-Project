"""Step-by-step verification of the MPI backend. Run from the repo root:

    python backends/mpi/verify_mpi.py            # full check (about 1 minute)
    python backends/mpi/verify_mpi.py --quick    # N=100 only

Checks, in order:
  1. toolchain  : numpy, mpi4py and mpiexec are available
  2. hello      : mpiexec really starts several ranks that can talk to each other
  3. correctness: nbody_mpi.py (both variants, P = 1, 2, 3, 4) reproduces the original
                  serial integrator to < 1e-10 (contract §C6). P = 3 exercises N % P != 0.
  4. team ref   : if data/ref/final_N{N}_steps10.csv exists, compare against it too
  5. json       : the result file follows contract schema v1 (§C4)
  6. errors     : a bad --variant fails loudly instead of running
Exit code 0 means everything passed.
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out" / "verify"
TOL = 1e-10
STEPS = 10
JSON_KEYS = {"schema_version": int, "backend": str, "variant": str, "lang": str, "n_bodies": int,
             "workers": int, "steps": int, "dt": float, "t_setup_s": float, "t_loop_s": float,
             "t_force_s": float, "t_comm_s": float, "t_transfer_s": float, "machine": str,
             "extra": dict}

results = []


def report(step, name, ok, detail=""):
    results.append((step, name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  —  {detail}" if detail else ""))
    return ok


def mpi_cmd(p, *script_args):
    return ["mpiexec", "-n", str(p), sys.executable, *map(str, script_args)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="only N=100")
    ap.add_argument("--procs", default="1,2,3,4", help="rank counts to test")
    a = ap.parse_args()
    sizes = [100] if a.quick else [100, 1000]
    procs = [int(p) for p in a.procs.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)

    print("\nStep 1 — toolchain")
    try:
        import numpy
        report(1, f"numpy {numpy.__version__}", True)
    except ImportError as exc:
        report(1, "numpy", False, str(exc))
        return finish()
    try:
        from mpi4py import MPI
        report(1, f"mpi4py {__import__('mpi4py').__version__} ({MPI.get_vendor()[0]})", True)
    except ImportError as exc:
        report(1, "mpi4py", False, f"{exc}  →  install system MPI, then pip install -r backends/mpi/requirements-mpi.txt")
        return finish()
    if not report(1, "mpiexec on PATH", shutil.which("mpiexec") is not None, shutil.which("mpiexec") or "not found"):
        return finish()

    print("\nStep 2 — hello across ranks")
    hello = ("from mpi4py import MPI; c = MPI.COMM_WORLD; "
             "s = c.allreduce(c.Get_rank(), op=MPI.SUM); "
             "print(c.Get_size(), s) if c.Get_rank() == 0 else None")
    r = subprocess.run(["mpiexec", "-n", "4", sys.executable, "-c", hello], capture_output=True, text=True, timeout=120)
    ok = r.returncode == 0 and r.stdout.split() == ["4", "6"]      # 0+1+2+3 = 6
    if not report(2, "4 ranks start and allreduce(rank) == 6", ok, (r.stdout + r.stderr).strip()[-200:]):
        return finish()

    sys.path.insert(0, str(HERE))
    import contract_io as nbio
    from test_data import compare_states, find_or_make_ic, reference_final
    print(f"\n  (I/O library in use: {nbio.SOURCE})")

    print(f"\nStep 3 — correctness vs original serial integrator ({STEPS} steps, tol {TOL:g})")
    last_json = None
    for n in sizes:
        ic, kind = find_or_make_ic(n)
        print(f"  N={n}: input {ic.relative_to(ROOT)} ({kind} IC)")
        _, _, p_ref, v_ref = reference_final(ic, STEPS)
        for variant in ("allgather", "master-worker"):
            for p in procs:
                final = OUT / f"final_N{n}_P{p}_{variant}.csv"
                js = OUT / f"result_N{n}_P{p}_{variant}.json"
                r = subprocess.run(mpi_cmd(p, HERE / "nbody_mpi.py", "--input", ic, "--steps", STEPS,
                                           "--variant", variant, "--machine", "verify",
                                           "--final-state", final, "--output-json", js),
                                   capture_output=True, text=True, timeout=600)
                if r.returncode != 0:
                    report(3, f"N={n} P={p} {variant}", False, r.stderr.strip()[-300:])
                    continue
                _, _, p_tst, v_tst = nbio.read_ic_csv(str(final))
                ep, ev = compare_states(p_tst, v_tst, p_ref, v_ref)
                report(3, f"N={n:<5} P={p} {variant:<13}", ep < TOL and ev < TOL,
                       f"max rel err pos {ep:.1e}, vel {ev:.1e}")
                last_json = js

        team_ref = ROOT / "data" / "ref" / f"final_N{n}_steps{STEPS}.csv"
        if kind == "team" and team_ref.exists():
            print(f"\nStep 4 — team reference {team_ref.relative_to(ROOT)}")
            _, _, p_t, v_t = nbio.read_ic_csv(str(team_ref))
            _, _, p_m, v_m = nbio.read_ic_csv(str(OUT / f"final_N{n}_P{procs[-1]}_allgather.csv"))
            ep, ev = compare_states(p_m, v_m, p_t, v_t)
            report(4, f"N={n} P={procs[-1]} vs team reference", ep < TOL and ev < TOL,
                   f"max rel err pos {ep:.1e}, vel {ev:.1e}")
    if not any(s == 4 for s, _, _ in results):
        print("\nStep 4 — team reference: skipped (no data/ic + data/ref files on this branch yet)")

    print("\nStep 5 — result JSON follows schema v1")
    if last_json and last_json.exists():
        doc = json.loads(last_json.read_text())
        missing = [k for k in JSON_KEYS if k not in doc]
        badtype = [k for k, t in JSON_KEYS.items() if k in doc and not isinstance(doc[k], t)]
        report(5, "all §C4 keys present with the right types", not missing and not badtype,
               f"missing {missing} wrong type {badtype}" if missing or badtype else last_json.name)
        report(5, "schema_version == 1 and backend == 'py-mpi'",
               doc.get("schema_version") == 1 and doc.get("backend") == "py-mpi")
        report(5, "workers == number of ranks, t_loop_s > 0",
               doc.get("workers") == procs[-1] and doc.get("t_loop_s", 0) > 0,
               f"workers={doc.get('workers')} t_loop_s={doc.get('t_loop_s')}")
    else:
        report(5, "result JSON written", False, "no successful run produced a JSON")

    print("\nStep 6 — errors fail loudly")
    ic, _ = find_or_make_ic(100)
    r = subprocess.run(mpi_cmd(2, HERE / "nbody_mpi.py", "--input", ic, "--variant", "nonsense"),
                       capture_output=True, text=True, timeout=120)
    report(6, "bad --variant exits non-zero", r.returncode != 0)
    r = subprocess.run(mpi_cmd(2, HERE / "nbody_mpi.py", "--input", OUT / "does_not_exist.csv"),
                       capture_output=True, text=True, timeout=120)
    report(6, "missing --input file exits non-zero", r.returncode != 0)
    return finish()


def finish():
    failed = [f"step {s}: {n.strip()}" for s, n, ok in results if not ok]
    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED {len(failed)} of {len(results)} checks:")
        for f in failed:
            print("  -", f)
    else:
        print(f"ALL {len(results)} CHECKS PASSED — the MPI backend is correct on this machine.")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
