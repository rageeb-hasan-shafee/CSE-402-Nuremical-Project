"""Summarise MPI sweep results: speed-up, efficiency, comm share, Amdahl fit.

    python backends/mpi/summarize_mpi.py --machine mansib-m4            # tables in the terminal
    python backends/mpi/summarize_mpi.py --machine mansib-m4 --plot     # + PNG graphs
    python backends/mpi/summarize_mpi.py --machine mansib-m4 --report   # + BENCHMARK_REPORT.md (implies --plot)

--report writes into the results folder itself:
    bench/results/py-mpi/<machine>/BENCHMARK_REPORT.md
    bench/results/py-mpi/<machine>/mpi_scaling_<variant>.png
    bench/results/py-mpi/<machine>/mpi_time_vs_n.png
Without --report, graphs go to backends/mpi/out/.

This is Mansib's own view of his results. The team-wide numbers and report
figures come from Fahad's bench/aggregate.py and bench/plots.py.

Definitions (docs/IMPLEMENTATION_GUIDE_ULTRA.md §7):
    time/step   median(t_loop_s) / steps over repeats r1.., warm-up r0 dropped
    S(P)        T(P=1) / T(P)            (same variant, same N, same machine)
    E(P)        S(P) / P
    comm %      t_comm_s / t_loop_s      (includes time spent waiting for slower ranks)
    Karp-Flatt  e = (1/S - 1/P) / (1 - 1/P)    (serial fraction measured at each P)
    Amdahl s    least-squares fit of S = 1 / (s + (1 - s)/P) over all P > 1
"""
import argparse
import collections
import datetime
import json
import math
import pathlib
import platform
import re
import statistics

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = re.compile(r"py-mpi_(?P<variant>.+)_N(?P<n>\d+)_P(?P<p>\d+)_r(?P<r>\d+)\.json$")
NAN = float("nan")


# ---------------------------------------------------------------- data
def load(machine):
    """-> {(variant, n, p): row}, plus run metadata."""
    runs = collections.defaultdict(list)
    for f in sorted((ROOT / "bench" / "results" / "py-mpi" / machine).glob("py-mpi_*.json")):
        m = NAME.search(f.name)
        if not m or m["r"] == "0":                       # skip warm-up
            continue
        runs[(m["variant"], int(m["n"]), int(m["p"]))].append(json.loads(f.read_text()))
    table, meta = {}, {"steps": set(), "dt": set(), "vendor": set(), "hosts": set(), "runs": 0}
    for key, docs in runs.items():
        steps = docs[0]["steps"]
        times = [d["t_loop_s"] / steps for d in docs]
        table[key] = dict(
            t=statistics.median(times),
            spread=(max(times) - min(times)) / statistics.median(times) if len(times) > 1 else 0.0,
            comm=statistics.median(d["t_comm_s"] / d["t_loop_s"] for d in docs if d["t_loop_s"] > 0),
            reps=len(docs))
        for d in docs:
            meta["runs"] += 1
            meta["steps"].add(d["steps"]); meta["dt"].add(d["dt"])
            ex = d.get("extra", {})
            meta["vendor"].add(ex.get("mpi_vendor", "?"))
            meta["hosts"].update(ex.get("processor_names", []))
    return table, meta


def amdahl_fit(points):
    """points: [(P, S)] with P > 1 -> serial fraction s (closed-form least squares)."""
    xs = [1 - 1 / p for p, _ in points]
    ys = [1 / s - 1 / p for p, s in points]
    den = sum(x * x for x in xs)
    return sum(x * y for x, y in zip(xs, ys)) / den if den else NAN


def analyse(table):
    """-> {(variant, n): {"rows": [...], "amdahl": s, "best": (P, S), "slower_at": [P...]}}"""
    series = collections.defaultdict(dict)
    for (variant, n, p), row in table.items():
        series[(variant, n)][p] = row
    out = {}
    for key in sorted(series):
        rows = series[key]
        base = rows.get(1)
        lines, pts = [], []
        for p in sorted(rows):
            r = rows[p]
            s = base["t"] / r["t"] if base else NAN
            kf = (1 / s - 1 / p) / (1 - 1 / p) if base and p > 1 else NAN
            if base and p > 1:
                pts.append((p, s))
            lines.append(dict(p=p, ms=1e3 * r["t"], s=s, e=s / p, comm=100 * r["comm"],
                              kf=kf, reps=r["reps"], spread=100 * r["spread"]))
        best = max(pts, key=lambda t: t[1]) if pts else (1, 1.0)
        out[key] = dict(rows=lines, amdahl=amdahl_fit(pts) if pts else NAN, best=best,
                        slower_at=[p for p, s in pts if s < 1.0], has_base=base is not None)
    return out


# ---------------------------------------------------------------- terminal
def print_tables(stats):
    for (variant, n), st in stats.items():
        print(f"\n{variant}  N={n}" + ("" if st["has_base"] else "   (no P=1 run → no speed-up)"))
        print(f"  {'P':>3} {'ms/step':>10} {'speed-up':>9} {'effic.':>7} {'comm %':>7} {'Karp-Flatt':>11} {'reps':>5}")
        for r in st["rows"]:
            print(f"  {r['p']:>3} {r['ms']:>10.3f} {r['s']:>9.2f} {r['e']:>7.2f} {r['comm']:>6.1f}% "
                  f"{r['kf']:>11.3f} {r['reps']:>5}")
        if st["has_base"] and len(st["rows"]) > 1:
            print(f"  Amdahl serial fraction s ≈ {st['amdahl']:.3f};  "
                  f"best speed-up {st['best'][1]:.2f}× at P={st['best'][0]}")


# ---------------------------------------------------------------- plots
def make_plots(stats, machine, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for variant in sorted({v for v, _ in stats}):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        pmax = 1
        for (v, n), st in stats.items():
            if v != variant or not st["has_base"]:
                continue
            ps = [r["p"] for r in st["rows"]]
            pmax = max(pmax, ps[-1])
            ax1.plot(ps, [r["s"] for r in st["rows"]], "o-", label=f"N={n}")
            ax2.plot(ps, [r["comm"] for r in st["rows"]], "o-", label=f"N={n}")
        ax1.plot([1, pmax], [1, pmax], "k--", lw=0.8, label="ideal")
        ax1.axhline(1.0, color="grey", lw=0.6, ls=":")
        ax1.set(xlabel="MPI ranks P", ylabel="speed-up S(P)", title=f"py-mpi {variant}: strong scaling")
        ax2.set(xlabel="MPI ranks P", ylabel="communication share of loop time (%)", title="why it flattens")
        for ax in (ax1, ax2):
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"machine: {machine}")
        fig.tight_layout()
        paths[variant] = out_dir / f"mpi_scaling_{variant}.png"
        fig.savefig(paths[variant], dpi=140)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    by_vp = collections.defaultdict(list)
    for (v, n), st in stats.items():
        for r in st["rows"]:
            by_vp[(v, r["p"])].append((n, r["ms"]))
    for (v, p), pts in sorted(by_vp.items()):
        if v != "allgather" and len({vv for vv, _ in by_vp}) > 1:
            continue
        pts.sort()
        if len(pts) > 1:
            ax.loglog([n for n, _ in pts], [ms for _, ms in pts], "o-", label=f"{v} P={p}")
    all_pts = sorted({n for (_, n) in stats})
    if len(all_pts) > 1:
        n0 = all_pts[-1]
        ref = [ms for (v, p), pts in by_vp.items() if p == 1 for n, ms in pts if n == n0]
        if ref:
            ax.loglog(all_pts, [ref[0] * (n / n0) ** 2 for n in all_pts], "k--", lw=0.8, label="∝ N² (guide)")
    ax.set(xlabel="bodies N", ylabel="time per step (ms)", title=f"time vs N ({machine})")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    paths["time_vs_n"] = out_dir / "mpi_time_vs_n.png"
    fig.savefig(paths["time_vs_n"], dpi=140)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------- report
def findings(stats):
    """Plain-language observations computed from the numbers."""
    lines = []
    for (variant, n), st in stats.items():
        if variant != "allgather" or not st["has_base"] or len(st["rows"]) < 2:
            continue
        p_best, s_best = st["best"]
        last = st["rows"][-1]
        txt = (f"**N = {n}:** best speed-up **{s_best:.2f}× at P = {p_best}**; at P = {last['p']} "
               f"it is {last['s']:.2f}× with {last['comm']:.0f}% of the time in communication.")
        if st["slower_at"]:
            txt += (f" With P = {', '.join(map(str, st['slower_at']))} it is **slower than one rank**: "
                    f"each rank has only ~{n // max(st['slower_at'])} bodies, so exchanging positions "
                    f"costs more than the saved work (the effect in Zhu 2020, Fig. 5).")
        elif p_best == last["p"]:
            txt += " Speed-up is still rising at the largest P, so this N could use more ranks."
        else:
            txt += f" Speed-up peaks at P = {p_best} and then falls, because communication is growing."
        lines.append(txt)
    variants = sorted({v for v, _ in stats})
    if {"allgather", "master-worker"} <= set(variants):
        ratios = []
        for (v, n), st in stats.items():
            if v != "allgather":
                continue
            other = stats.get(("master-worker", n))
            if not other:
                continue
            a = {r["p"]: r["ms"] for r in st["rows"]}
            b = {r["p"]: r["ms"] for r in other["rows"]}
            ratios += [b[p] / a[p] for p in a if p in b and p > 1]
        if ratios:
            med = statistics.median(ratios)
            lines.append(f"**master-worker vs allgather:** master-worker takes {med:.2f}× the time of "
                         f"allgather (median over all N and P > 1). "
                         + ("On one machine the two are close; the gap should grow on a cluster, where "
                            "rank 0's double job crosses the network." if med < 1.1 else
                            "The paper's two-step design (Gatherv + Bcast) is measurably slower."))
    return lines


def write_report(stats, meta, machine, plots, path):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    hosts = sorted(meta["hosts"]) or ["?"]
    L = [f"# MPI benchmark report: `{machine}`", "",
         f"Generated {ts} by `backends/mpi/summarize_mpi.py` from {meta['runs']} timed runs "
         f"(warm-up runs excluded).", "",
         "## Setup", "",
         "| | |", "|---|---|",
         f"| Backend | `py-mpi` (mpi4py, velocity Verlet, block decomposition, one exchange per step) |",
         f"| Machine tag | `{machine}` |",
         f"| Host(s) | {', '.join(hosts)} ({len(hosts)} node{'s' if len(hosts) > 1 else ''}) |",
         f"| OS / Python | {platform.system()} {platform.release()} / Python {platform.python_version()} (report-generation machine) |",
         f"| MPI | {', '.join(sorted(meta['vendor']))} |",
         f"| Steps per run / dt | {', '.join(map(str, sorted(meta['steps'])))} / {', '.join(map(str, sorted(meta['dt'])))} day |",
         "| Timing | median of repeats, `t_loop_s / steps`, slowest rank; warm-up r0 dropped |", "",
         "## Key findings", ""]
    L += [f"- {x}" for x in findings(stats)] or ["- (need P = 1 runs to compute speed-ups)"]
    L += ["", "## Graphs", ""]
    for key, p in plots.items():
        L.append(f"![{key}]({p.name})")
        L.append("")
    L += ["## Tables", "",
          "S = speed-up vs P = 1 (same variant, same N); E = S/P; comm % includes waiting for slower ranks; "
          "spread = (max − min)/median over repeats (run-to-run noise).", ""]
    for (variant, n), st in stats.items():
        L.append(f"### {variant}, N = {n}")
        L.append("")
        L.append("| P | ms/step | speed-up S | efficiency E | comm % | Karp–Flatt | repeats | spread |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in st["rows"]:
            kf = "" if math.isnan(r["kf"]) else f"{r['kf']:.3f}"
            L.append(f"| {r['p']} | {r['ms']:.3f} | {r['s']:.2f} | {r['e']:.2f} | {r['comm']:.1f} | "
                     f"{kf} | {r['reps']} | {r['spread']:.0f}% |")
        if st["has_base"] and len(st["rows"]) > 1:
            L.append("")
            L.append(f"Amdahl fit: serial fraction **s ≈ {st['amdahl']:.3f}** "
                     f"(maximum possible speed-up ≈ {1 / st['amdahl']:.1f}×)" if st["amdahl"] > 0 else
                     "Amdahl fit: s ≤ 0 (scaling at or above ideal; too few points or noise)")
        L.append("")
    L += ["## How to read this", "",
          "- **Speed-up S(P)** = time with 1 rank ÷ time with P ranks. Ideal is S = P.",
          "- **Efficiency E** = S/P. 1.0 means no wasted ranks.",
          "- **comm %** is the share of loop time inside the position exchange, including waiting for the "
          "slowest rank at that synchronisation point. When it grows, speed-up flattens.",
          "- **Karp–Flatt** is the serial fraction measured at each P. If it rises with P, overhead "
          "(communication, synchronisation) is growing.",
          "- **Amdahl s**: if a fraction s of the work cannot be parallelised, the speed-up can never exceed 1/s.", "",
          "Reproduce: `python backends/mpi/sweep_local.py --machine " + machine +
          "` then `python backends/mpi/summarize_mpi.py --machine " + machine + " --report`.", ""]
    path.write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", required=True)
    ap.add_argument("--plot", action="store_true", help="write PNG graphs to backends/mpi/out/")
    ap.add_argument("--report", action="store_true", help="write BENCHMARK_REPORT.md + graphs into the results folder")
    a = ap.parse_args()
    table, meta = load(a.machine)
    if not table:
        print(f"no results under bench/results/py-mpi/{a.machine}/ — run sweep_local.py first")
        return
    stats = analyse(table)
    print_tables(stats)
    if a.report:
        res_dir = ROOT / "bench" / "results" / "py-mpi" / a.machine
        plots = make_plots(stats, a.machine, res_dir)
        report = res_dir / "BENCHMARK_REPORT.md"
        write_report(stats, meta, a.machine, plots, report)
        print(f"\nreport: {report.relative_to(ROOT)}")
        for p in plots.values():
            print(f"graph:  {p.relative_to(ROOT)}")
    elif a.plot:
        for p in make_plots(stats, a.machine, HERE / "out" / a.machine).values():
            print(f"saved {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
