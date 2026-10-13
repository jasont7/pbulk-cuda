"""Turns measured results into the LaTeX fragments and the timing plot that report/ir1.tex includes.

    python report/fill_results.py                                  # no notebook results yet: Oct 6 numbers + TBD
    python report/fill_results.py report/data/ir1_results.json     # after the Colab notebook has been run

Rules this script enforces:
  * a number is only printed if it is in RESULTS.md (via data/oct6_results.json) or in a results file
    produced on a real device (meta.backend == "cuda"). Anything else stays a visible TBD.
  * speedups are truncated, never rounded up.
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "figures"))

TBD = r"\TBD{run notebook}"
SIZE_ORDER = ["small", "medium", "large"]


def med(x):
    return float(np.median(np.asarray(x, dtype=np.float64).ravel()))


def fmt_ms(v):
    """Milliseconds in the style of RESULTS.md: 13,935 / 246.5 / 25.5 / 4.90"""
    if v >= 1000:
        return f"{v:,.0f}"
    return f"{v:.1f}" if v >= 10 else f"{v:.2f}"


def fmt_x(v):
    """A speedup, truncated so it is never overstated: 16.8 / 652 / 9,186"""
    if v >= 100:
        return f"{math.floor(v):,d}"
    return f"{math.floor(v * 10) / 10:.1f}"


def fmt_sci(v):
    if v == 0:
        return "0"
    m, e = f"{v:.1e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def fmt_count(v):
    """16,000,000 -> 16.0 M"""
    return f"{v / 1e9:.2f} G" if v >= 1e9 else f"{v / 1e6:.1f} M"


def num(s):
    return float(str(s).replace(",", ""))


def load_results(path):
    r = json.loads(pathlib.Path(path).read_text())
    if r.get("schema") != 1:
        raise SystemExit(f"{path}: unknown schema {r.get('schema')}")
    if r["meta"]["backend"] != "cuda":
        raise SystemExit(f"{path}: produced by backend '{r['meta']['backend']}', not a GPU. "
                         "Refusing to put these numbers in the report.")
    return r


# ---------------------------------------------------------------- tables

def instance_rows(counts):
    rows = []
    for name in SIZE_ORDER:
        c = counts[name]
        rows.append(f"{name} & {c['genes']:,d} $\\times$ {c['cells']:,d} & {fmt_count(c['nnz'])} & "
                    f"{fmt_count(c['terms'])} & {c['terms'] / c['nnz']:.1f} & "
                    f"{c['gather_launch']['threads'] / 1e6:.2f} M \\\\")
    return "\n".join(rows)


def kernel_rows_oct6(o):
    rows = []
    for name in ("medium", "large"):
        s = o["sizes"][name]
        rows.append(f"{name} & {s['cpu_loop_ms'] or 'not run'} & {s['cpu_batched_ms']} & {TBD_SHORT} & "
                    f"{s['scatter_ms']} & \\textbf{{{s['gather_ms']}}} \\\\")
    return "\n".join(rows)


TBD_SHORT = r"\TBD{}"


def kernel_rows(r):
    rows = []
    for name in SIZE_ORDER:
        k = r.get("kernel", {}).get(name)
        if not k:
            rows.append(f"{name} & " + " & ".join([TBD_SHORT] * 5) + r" \\")
            continue
        loop = fmt_ms(med(k["cpu_loop_ms"])) if k.get("cpu_loop_ms") else "not run"
        cpp = fmt_ms(med(k["cpu_cpp_ms"])) if k.get("cpu_cpp_ms") else "n/a"
        rows.append(f"{name} & {loop} & {fmt_ms(med(k['cpu_batched_ms']))} & {cpp} & "
                    f"{fmt_ms(med(k['scatter_ms']))} & \\textbf{{{fmt_ms(med(k['gather_ms']))}}} \\\\")
    return "\n".join(rows)


def speedup_rows_oct6(o):
    m, l = o["sizes"]["medium"], o["sizes"]["large"]
    x = "$\\times$"
    return "\n".join([
        f"gather vs scatter (ablation) & not run & {m['gather_vs_scatter']}{x} & {l['gather_vs_scatter']}{x} \\\\",
        f"gather vs CPU scipy product & not run & {m['gather_vs_batched']}{x} & {l['gather_vs_batched']}{x} \\\\",
        f"gather vs CPU loop & not run & {m['gather_vs_loop']}{x} & not run \\\\",
        f"gather vs CPU C++ (Alg. 1) & {TBD_SHORT} & {TBD_SHORT} & {TBD_SHORT} \\\\",
    ])


def speedup_rows(r):
    def cell(name, base):
        k = r.get("kernel", {}).get(name)
        if not k or not k.get(base):
            return "not run" if k else TBD
        return fmt_x(med(k[base]) / med(k["gather_ms"])) + r"$\times$"
    def trial_range(name):
        k = r.get("kernel", {}).get(name)
        if not k:
            return TBD
        ratios = [med(s) / med(g) for s, g in zip(k["scatter_ms"], k["gather_ms"])]
        return f"{fmt_x(min(ratios))} to {fmt_x(max(ratios))}"
    rows = []
    for label, base in (("gather vs scatter (ablation)", "scatter_ms"), ("gather vs CPU scipy product", "cpu_batched_ms"),
                        ("gather vs CPU C++ (Alg. 1)", "cpu_cpp_ms"), ("gather vs CPU loop", "cpu_loop_ms")):
        rows.append(f"{label} & " + " & ".join(cell(n, base) for n in SIZE_ORDER) + r" \\")
    rows.append("ablation, range over trials & " + " & ".join(trial_range(n) for n in SIZE_ORDER) + r" \\")
    return "\n".join(rows)


REGIMES = ["one warp", "one block, several warps", "several blocks"]


def correctness_rows(r):
    c = r.get("correctness") if r else None
    rows = []
    for reg in REGIMES:
        cells = []
        for k in ("scatter", "gather"):
            if not c:
                cells.append(TBD_SHORT)
                continue
            hit = [x[k] for x in c["cases"] if x[k]["regime"] == reg]
            cells.append(f"{sum(x['ok'] for x in hit)} of {len(hit)}")
        rows.append(f"{reg} & {cells[0]} & {cells[1]} \\\\")
    rows.append(r"\midrule")
    if c:
        ints = [x for x in c["cases"] if x["integer"]]
        rows.append("integer data, bit-exact & " + " & ".join(
            f"{sum(x[k]['exact'] for x in ints)} of {len(ints)}" for k in ("scatter", "gather")) + r" \\")
        rows.append("worst rel. error (float data) & " + " & ".join(
            fmt_sci(max(x[k]["max_rel_err"] for x in c["cases"] if not x["integer"])) for k in ("scatter", "gather"))
            + r" \\")
    else:
        rows.append(f"integer data, bit-exact & {TBD_SHORT} & {TBD_SHORT} \\\\")
        rows.append(f"worst rel. error (float data) & {TBD_SHORT} & {TBD_SHORT} \\\\")
    return "\n".join(rows)


def e2e_rows(r):
    e = r.get("end_to_end") if r else None
    if not e:
        return "\n".join([f"scatter & {' & '.join([TBD_SHORT] * 5)} \\\\", f"gather & {' & '.join([TBD_SHORT] * 5)} \\\\",
                          r"\midrule", f"CPU C++ (Alg. 1) & & & & & {TBD_SHORT} \\\\",
                          f"CPU scipy product & & & & & {TBD_SHORT} \\\\"])
    rows = []
    for k in ("scatter", "gather"):
        p = [fmt_ms(med(e[k][s])) for s in e["stages"]]
        rows.append(f"{k} & {' & '.join(p)} & \\textbf{{{fmt_ms(med(e[k]['total_ms']))}}} \\\\")
    rows.append(r"\midrule")
    cpp = fmt_ms(med(e["cpu_cpp_ms"])) if e.get("cpu_cpp_ms") else "n/a"
    rows.append(f"CPU C++ (Alg. 1) & & & & & {cpp} \\\\")
    rows.append(f"CPU scipy product & & & & & {fmt_ms(med(e['cpu_batched_ms']))} \\\\")
    return "\n".join(rows)


NCU_METRICS = [("Achieved Occupancy", "achieved occupancy (Nsight)"),
               ("Compute (SM) Throughput", "SM throughput (Nsight)"),
               ("Memory Throughput", "memory throughput (Nsight)")]


def profile_rows(r):
    p = r.get("profile") if r else None
    if not p:
        return "\n".join(f"{label} & {TBD_SHORT} & {TBD_SHORT} \\\\" for label in (
            "GPU utilisation (nvidia-smi)", "memory utilisation (nvidia-smi)", "registers per thread (ptxas)",
            "stack frame, bytes (ptxas)", "achieved occupancy (Nsight)"))
    rows = []
    smi = p.get("smi", {})
    for field, label in (("utilization.gpu", "GPU utilisation (nvidia-smi)"),
                         ("utilization.memory", "memory utilisation (nvidia-smi)")):
        cells = [f"{smi[k][field]['median']:.0f}\\%" if smi.get(k, {}).get("available") and field in smi[k]
                 else "not available" for k in ("scatter", "gather")]
        rows.append(f"{label} & {cells[0]} & {cells[1]} \\\\")
    ru = p.get("resource_usage", {}).get("kernels", {})
    for key, label in (("registers", "registers per thread (ptxas)"), ("stack_bytes", "stack frame, bytes (ptxas)")):
        cells = [str(ru[k][key]) if key in ru.get(k, {}) else "not available" for k in ("pb_scatter", "pb_gather")]
        rows.append(f"{label} & {cells[0]} & {cells[1]} \\\\")
    ncu = p.get("ncu", {})
    if ncu.get("available"):
        for metric, label in NCU_METRICS:
            cells = []
            for k in ("pb_scatter", "pb_gather"):
                m = ncu["kernels"].get(k, {}).get(metric)
                cells.append(f"{m['value']}{(m['unit'] or '').replace('%', chr(92) + '%')}" if m else "not reported")
            rows.append(f"{label} & {cells[0]} & {cells[1]} \\\\")
    else:
        rows.append(r"Nsight Compute counters & \multicolumn{2}{c}{not available on this runtime} \\")
    return "\n".join(rows)


# ---------------------------------------------------------------- inline numbers

def macros_oct6(o):
    m, l = o["sizes"]["medium"], o["sizes"]["large"]
    gbps = 76.8e6 * 8 / (num(l["gather_ms"]) * 1e-3) / 1e9          # col + val bytes of the large instance per second
    return {
        "ResSource": f"{o['date']} run on a Colab {o['gpu']} (CuPy {o['cupy']}), one session. GPU times are medians "
                     f"of {o['gpu_repeats']} runs. CPU times are the best of 3 runs (the loop ran once)",
        "ResShort": f"{o['date']} run",
        "ResEnv": f"A Colab {o['gpu']} with CuPy {o['cupy']}. The CPU model of that session was not recorded.",
        "GatherMsMedium": m["gather_ms"], "GatherMsLarge": l["gather_ms"],
        "ScatterMsMedium": m["scatter_ms"], "ScatterMsLarge": l["scatter_ms"],
        "ScipyMsMedium": m["cpu_batched_ms"], "ScipyMsLarge": l["cpu_batched_ms"],
        "LoopMsMedium": m["cpu_loop_ms"],
        "AblationMedium": m["gather_vs_scatter"], "AblationLarge": l["gather_vs_scatter"],
        "VsScipyMedium": m["gather_vs_batched"], "VsScipyLarge": l["gather_vs_batched"],
        "VsLoopMedium": m["gather_vs_loop"],
        "VsCppMedium": TBD, "VsCppLarge": TBD,
        "FullSizeVerdict": f"On {o['date'].replace('Oct', 'October')}, on the GPU, both kernels matched the CPU reference "
                           f"on the medium and large instances. The largest relative errors were {m['scatter_err']} "
                           f"and {l['scatter_err']} for scatter and {m['gather_err']} and {l['gather_err']} for gather.",
        "ResCaveat": "The October 6 numbers come from one session, with medians of 10 runs on the GPU but the "
                     "best of 3 runs, or a single run for the loop, on the CPU.",
        "GatherGBps": f"{math.floor(gbps):d}",
        "KernelRows": kernel_rows_oct6(o), "SpeedupRows": speedup_rows_oct6(o),
        "EndToEndVerdict": TBD, "ProfileVerdict": TBD, "CorrectnessVerdict": TBD,
        "NotebookStatus": r"\TBD{The notebook has not been run yet. Tables~\ref{tab:correct}, \ref{tab:e2e} and "
                          r"\ref{tab:profile} are placeholders until it has.}",
    }


def macros_results(r, o):
    k = r.get("kernel", {})
    meta = r["meta"]
    def ms(size, key):
        return fmt_ms(med(k[size][key])) if k.get(size, {}).get(key) else TBD
    def ratio(size, key):
        return fmt_x(med(k[size][key]) / med(k[size]["gather_ms"])) if k.get(size, {}).get(key) else TBD
    gpu = meta["gpu"]
    n_trials = next(iter(k.values()))["trials"] if k else 0
    n_rep = next(iter(k.values()))["repeats"] if k else 0
    out = {
        "ResSource": f"run of {meta['created_utc'][:10]} on a Colab {gpu.get('name')} (CuPy {gpu.get('cupy')}). GPU times "
                     f"are medians over {n_trials} trials of {n_rep} runs. CPU times are medians of 3 runs "
                     f"(the loop ran once)",
        "ResShort": f"{meta['created_utc'][:10]} run",
        "ResEnv": f"A Colab {gpu.get('name')} with {gpu.get('multiProcessorCount', '?')} SMs, CuPy {gpu.get('cupy')}. "
                  f"CPU: {tex_escape(meta.get('cpu') or 'unknown')}, baselines on one thread.",
        "GatherMsMedium": ms("medium", "gather_ms"), "GatherMsLarge": ms("large", "gather_ms"),
        "ScatterMsMedium": ms("medium", "scatter_ms"), "ScatterMsLarge": ms("large", "scatter_ms"),
        "ScipyMsMedium": ms("medium", "cpu_batched_ms"), "ScipyMsLarge": ms("large", "cpu_batched_ms"),
        "LoopMsMedium": ms("medium", "cpu_loop_ms"),
        "AblationMedium": ratio("medium", "scatter_ms"), "AblationLarge": ratio("large", "scatter_ms"),
        "VsScipyMedium": ratio("medium", "cpu_batched_ms"), "VsScipyLarge": ratio("large", "cpu_batched_ms"),
        "VsLoopMedium": ratio("medium", "cpu_loop_ms"),
        "VsCppMedium": ratio("medium", "cpu_cpp_ms"), "VsCppLarge": ratio("large", "cpu_cpp_ms"),
        "KernelRows": kernel_rows(r), "SpeedupRows": speedup_rows(r),
        "NotebookStatus": "",
    }
    out["ResCaveat"] = (f"All numbers come from one Colab session. Each GPU time is the median of {n_trials} trials "
                        f"of {n_rep} runs and each CPU time the median of 3 runs, except the loop, which ran once.")
    if k:
        bad = [f"{kk} on {s}" for s in k for kk in ("scatter", "gather") if not k[s][f"{kk}_ok"]]
        worst = {kk: fmt_sci_plain(max(k[s][f"{kk}_max_rel_err"] for s in k)) for kk in ("scatter", "gather")}
        out["FullSizeVerdict"] = (
            rf"\TBD{{These full-size runs did NOT match the CPU reference: {', '.join(bad)}.}}" if bad else
            f"On the GPU, both kernels matched the CPU reference on the {', '.join(k)} instances. The largest "
            f"relative error was {worst['scatter']} for scatter and {worst['gather']} for gather.")
    else:
        out["FullSizeVerdict"] = TBD
    if "large" in k:
        w = k["large"]["work"]
        out["GatherGBps"] = f"{math.floor(w['nnz'] * 8 / (med(k['large']['gather_ms']) * 1e-3) / 1e9):d}"
    else:
        out["GatherGBps"] = TBD

    e = r.get("end_to_end")
    if e:
        parts = []
        for name in ("scatter", "gather"):
            total = med(e[name]["total_ms"])
            for base, label in (("cpu_cpp_ms", "the C++ baseline"), ("cpu_batched_ms", "the scipy product")):
                if e.get(base):
                    b = med(e[base])
                    how = f"{fmt_x(b / total)}$\\times$ faster than" if total < b else \
                        f"{fmt_x(total / b)}$\\times$ slower than"
                    parts.append(f"{name} is {how} {label}")
        out["EndToEndVerdict"] = "End to end, " + ", ".join(parts) + "."
    else:
        out["EndToEndVerdict"] = TBD
    c = r.get("correctness")
    if c:
        n_ok = sum(x[kk]["ok"] for x in c["cases"] for kk in ("scatter", "gather"))
        out["CorrectnessVerdict"] = (f"On the GPU, {n_ok} of {2 * c['n_cases']} kernel runs over {c['n_cases']} instances "
                                     f"match the float64 reference.")
    else:
        out["CorrectnessVerdict"] = TBD
    p = r.get("profile")
    if p and any(v.get("available") for v in p.get("smi", {}).values()):
        out["ProfileVerdict"] = f"Measured on the {p['size']} instance."
    else:
        out["ProfileVerdict"] = TBD
    notes = prose_checks(r)
    if notes:
        out["NotebookStatus"] = " ".join(rf"\TBD{{{n}}}" for n in notes)
    return out


def prose_checks(r):
    """Sentences of ir1.tex that assume something about the results. Each message becomes a red TBD."""
    k, notes = r.get("kernel", {}), []
    if r.get("errors"):
        notes.append("These notebook stages failed and must be rerun: "
                     + ", ".join(tex_escape(x["stage"]) for x in r["errors"]) + ".")
    for s in k:
        for base in ("scatter_ms", "cpu_batched_ms", "cpu_cpp_ms", "cpu_loop_ms"):
            if k[s].get(base) and med(k[s][base]) <= med(k[s]["gather_ms"]):
                notes.append(f"Section 4.3 says gather is faster, but it is not faster than {tex_escape(base)} "
                             f"on {s}. Rewrite that paragraph.")
    if "medium" in k and "large" in k:
        ratio = {s: med(k[s]["scatter_ms"]) / med(k[s]["gather_ms"]) for s in ("medium", "large")}
        if ratio["large"] <= ratio["medium"]:
            notes.append("Section 4.3 says the gap between the kernels widens with size. It did not in this run. "
                         "Rewrite that sentence and future work item 2.")
    c = r.get("correctness")
    if c and not c["all_ok"]:
        notes.append("Some correctness instances failed on the GPU. Section 4.2 must say so.")
    return notes


def fmt_sci_plain(v):
    m, e = f"{v:.1e}".split("e")
    return f"{m}e{int(e)}"


def tex_escape(s):
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def count_macros(counts):
    m = counts["medium"]
    l = counts["large"]
    return {
        "InstanceRows": instance_rows(counts),
        "TermsMedium": fmt_count(m["terms"]), "NnzMedium": fmt_count(m["nnz"]),
        "SelPerCellMedium": f"{m['terms'] / m['nnz']:.1f}",
        "GatherAddsMedium": fmt_count(m["gather_masked_adds"]),
        "WorkRatioMedium": f"{m['gather_masked_adds'] / m['terms']:.2f}",
        "WorkRatioLarge": f"{l['gather_masked_adds'] / l['terms']:.2f}",
        "ShuffleAddsMedium": fmt_count(m["segments"] * 32 * 32 * 5),
        "SegLenMedium": f"{m['nnz'] / m['segments']:.0f}", "SegLenLarge": f"{l['nnz'] / l['segments']:.0f}",
        "ThreadsMedium": f"{m['gather_launch']['threads']:,d}", "ThreadsLarge": f"{l['gather_launch']['threads']:,d}",
        "ScatterThreads": f"{m['scatter_launch']['threads']:,d}",
    }


# ---------------------------------------------------------------- figure

def timing_panels_oct6(o):
    panels = []
    for name in ("medium", "large"):
        s = o["sizes"][name]
        data = {key: (num(s[src]), None, None, s[src] + " ms") for key, src in (
            ("cpu_loop", "cpu_loop_ms"), ("cpu_batched", "cpu_batched_ms"), ("scatter", "scatter_ms"),
            ("gather", "gather_ms")) if s[src]}
        panels.append((f"{name}: {s['genes']} genes x {s['cells']} cells, {s['nnz']} nonzeros", data))
    return panels


def timing_panels(r):
    panels = []
    for name in SIZE_ORDER:
        k = r.get("kernel", {}).get(name)
        if not k:
            continue
        data = {}
        for key, src in (("cpu_loop", "cpu_loop_ms"), ("cpu_batched", "cpu_batched_ms"), ("cpu_cpp", "cpu_cpp_ms"),
                         ("scatter", "scatter_ms"), ("gather", "gather_ms")):
            if k.get(src):
                v = np.asarray(k[src], dtype=np.float64).ravel()
                data[key] = (float(np.median(v)), float(v.min()), float(v.max()), fmt_ms(float(np.median(v))) + " ms")
        panels.append((f"{name}: {k['genes']:,d} genes x {k['cells']:,d} cells, {fmt_count(k['work']['nnz'])} nonzeros",
                       data))
    return panels


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", help="results JSON from the notebook (optional)")
    ap.add_argument("--outdir", default=str(HERE), help="where generated/ and figures/ go (tests use a temp dir)")
    a = ap.parse_args(argv)

    out = pathlib.Path(a.outdir)
    (out / "generated").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    o = json.loads((HERE / "data" / "oct6_results.json").read_text())
    counts = json.loads((HERE / "data" / "work_counts.json").read_text())
    r = load_results(a.results) if a.results else None

    macros = macros_results(r, o) if r else macros_oct6(o)
    macros.update(count_macros(counts))
    macros.update(CorrectnessRows=correctness_rows(r), EndToEndRows=e2e_rows(r), ProfileRows=profile_rows(r))
    lines = ["% Generated by report/fill_results.py. Do not edit by hand.",
             f"% Source: {'notebook results ' + str(a.results) if r else 'RESULTS.md (Oct 6), notebook not run yet'}"]
    for name, body in macros.items():
        lines.append(f"\\newcommand{{\\{name}}}{{%\n{body}}}")
    (out / "generated" / "results.tex").write_text("\n".join(lines) + "\n")

    import fig_timing
    fig_timing.draw(timing_panels(r) if r else timing_panels_oct6(o), out / "figures" / "fig_timing.pdf")
    for note in (prose_checks(r) if r else []):
        print("CHECK THE TEXT:", note)
    print(f"wrote {out / 'generated' / 'results.tex'} and {out / 'figures' / 'fig_timing.pdf'} "
          f"({'notebook results' if r else 'Oct 6 numbers, notebook parts are TBD'})")
    return macros


if __name__ == "__main__":
    main()
