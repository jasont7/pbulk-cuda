"""Tables and the figure of the IR2 experiments section, from report/data/ir2_results.json.

    python report/ir2_tables.py          # writes report/generated/ir2_tables.tex and figures/ir2_*.pdf
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "data" / "ir2_results.json"
OUT = HERE / "generated" / "ir2_tables.tex"
FIG = HERE / "figures"
KNAMES = {"scatter": "scatter (IR1 v1)", "gather": "gather (IR1 v2)", "gather_xr": "gather\\_xr (3b)",
          "gather_fm": "gather\\_fm (3)", "gather_cs": "gather\\_cs (2)", "cusparse": "cuSPARSE SpGEMM"}


def med(x):
    return float(np.median(np.ravel(x)))


def f(x, d=2):
    return f"{x:,.{d}f}"


def ncu_val(k, *names):
    for n in names:
        if n in k:
            v = k[n]["value"].replace(",", "")
            try:
                return float(v)
            except ValueError:
                return None
    return None


def main():
    r = json.loads(RES.read_text())
    if r["meta"]["backend"] != "cuda" and "--allow-fake" not in sys.argv:
        raise SystemExit("results not from a real GPU")
    K = r["kernel"]
    sizes = [s for s in ("small", "medium", "large") if s in K]
    L = []

    # ---- layout table (idea 1) ----
    L += [r"\begin{table}[t]", r"\caption{Idea 1: building the donor layout. Host is the IR1 NumPy/SciPy code on one core,"
          r" GPU is the same arrays built with CuPy and cuSPARSE (median of 5, synchronised). Identical: all four arrays"
          r" equal the host ones bit for bit.}", r"\label{tab:layout}", r"\small",
          r"\begin{tabular}{@{}lrrrc@{}}", r"\toprule",
          r"Size (nnz) & Host ms & GPU ms & Speedup & Identical \\", r"\midrule"]
    for s in sizes:
        k = K[s]
        h, g = k["layout_host_ms"], med(k["layout_gpu_ms"])
        L.append(f"{s} ({k['nnz']/1e6:.1f} M) & {f(h,1)} & {f(g,2)} & {f(h/g,0)}$\\times$ & "
                 f"{'yes' if all(k['layout_identical'].values()) else 'NO'} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # ---- kernel table ----
    L += [r"\begin{table*}[t]", r"\caption{Kernel time in ms (CUDA events, median of 3 trials of 10, includes the row"
          r" normalisation) and speedup over the IR1 gather kernel. Max relative error is against the float64 reference"
          r" after normalisation. 128-thread blocks.}", r"\label{tab:kernels}", r"\small",
          r"\begin{tabular}{@{}l" + "rr" * len(sizes) + r"r@{}}", r"\toprule",
          "Kernel & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{s} ({K[s]['nnz']/1e6:.1f} M nnz)}}" for s in sizes)
          + r" & Max rel.\ err \\",
          " & " + " & ".join(["ms & vs gather"] * len(sizes)) + r" & \\", r"\midrule"]
    for kn in ("scatter", "gather", "gather_xr", "gather_fm", "gather_cs", "cusparse"):
        cells, errs = [], []
        for s in sizes:
            k = K[s]
            if f"{kn}_ms" in k and k[f"{kn}_ms"]:
                t = med(k[f"{kn}_ms"]); base = med(k["gather_ms"])
                cells += [f(t, 2 if t < 100 else 1), f"{base/t:.2f}$\\times$"]
                errs.append(k[f"{kn}_max_rel_err"])
            else:
                cells += ["--", "--"]
        L.append(f"{KNAMES[kn]} & " + " & ".join(cells) + f" & {max(errs):.1e} \\\\" if errs else "")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]

    # ---- work counts for idea 2 ----
    L += [r"\begin{table}[t]", r"\caption{Idea 2, counted on the host from the actual layouts: adds per nonzero, and lane"
          r" steps per warp loop iteration (the warp waits for its slowest lane).}", r"\label{tab:work}", r"\small",
          r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", r" & " + " & ".join(sizes) + r" \\", r"\midrule"]
    rows = [("Set bits per cell (no flip)", lambda w: f"{w['mean_unflipped_set_bits']:.1f}"),
            ("Set bits after the flip", lambda w: f"{w['mean_set_bits']:.2f}"),
            ("Flipped pseudo-bulks (of 900)", lambda w: f"{w['flipped_pseudo_bulks']}"),
            ("Adds per nonzero, gather", lambda w: "32"),
            ("Adds per nonzero, gather\\_cs", lambda w: f"{w['complement_adds']/w['nnz']:.2f}"),
            ("Warp-max set bits per step", lambda w: f"{w['mean_warp_max_set_bits']:.2f}"),
            ("Lane-step ratio gather / cs", lambda w: f"{w['lane_steps_baseline']/w['lane_steps_cs']:.2f}$\\times$")]
    for name, fn in rows:
        L.append(name + " & " + " & ".join(fn(K[s]["work_cs"]) for s in sizes) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # ---- end to end ----
    E = r.get("end_to_end", {})
    stages = ("host_prep_ms", "h2d_ms", "layout_gpu_ms", "kernel_ms", "d2h_ms")
    pnames = {"scatter": "scatter", "gather_ir1": "gather, host layout (IR1)",
              "gather_gpu_layout": "gather, GPU layout", "gather_fm_gpu_layout": "gather\\_fm, GPU layout"}
    for s in ("medium", "large"):
        if s not in E:
            continue
        L += [r"\begin{table}[t]", f"\\caption{{End to end on the {s} instance, ms (median of 5). Host prep is the IR1"
              r" layout build or, for the GPU paths, the donor codes. GPU layout includes the fused-mask gather.}",
              f"\\label{{tab:e2e-{s}}}", r"\small", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
              r"Path & Host & H2D & Layout & Kernel & D2H & Total \\", r"\midrule"]
        for p, nm in pnames.items():
            e = E[s].get(p)
            if not e or "error" in e:
                continue
            L.append(nm + " & " + " & ".join(f(med(e[st]), 1) for st in stages) + f" & \\textbf{{{f(med(e['total_ms']),1)}}} \\\\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # ---- ncu ----
    N = r.get("ncu", {})
    if N.get("available"):
        kk = [k for k in ("pb_gather", "pb_gather_xr", "pb_gather_fm", "pb_gather_cs") if k in N["kernels"]]
        metrics = [("Duration (us)", ("Duration",), lambda v: v / 1e3, 0),
                   ("SM throughput (\\%)", ("Compute (SM) Throughput",), None, 1),
                   ("L1/TEX throughput (\\%)", ("L1/TEX Cache Throughput",), None, 1),
                   ("DRAM throughput (\\%)", ("DRAM Throughput",), None, 1),
                   ("Instructions (M)", ("smsp__inst_executed.sum", "Executed Instructions"), lambda v: v / 1e6, 1),
                   ("Active threads per instr.", ("smsp__thread_inst_executed_per_inst_executed.ratio",
                                                  "Avg. Active Threads Per Warp"), None, 1),
                   ("Warp cycles per issue", ("Warp Cycles Per Issued Instruction",), None, 1),
                   ("Registers per thread", ("Registers Per Thread",), None, 0),
                   ("Achieved occupancy (\\%)", ("Achieved Occupancy",), None, 1),
                   ("Shared bank conflicts (M)", ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",), lambda v: v / 1e6, 2),
                   ("Global load instr. (M)", ("smsp__sass_inst_executed_op_global_ld.sum",), lambda v: v / 1e6, 2),
                   ("L1/TEX bytes (MB)", ("l1tex__t_bytes.sum",), lambda v: v / 1e6, 0),
                   ("DRAM bytes read (MB)", ("dram__bytes_read.sum",), lambda v: v / 1e6, 0)]
        L += [r"\begin{table}[t]", r"\caption{Nsight Compute, medium instance, one launch each (SM clock locked by"
              r" Nsight, so durations exceed the event timings).}", r"\label{tab:ncu2}", r"\footnotesize",
              r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}l" + "r" * len(kk) + r"@{}}", r"\toprule",
              "Metric & " + " & ".join(k.replace("pb_", "").replace("_", "\\_") for k in kk) + r" \\", r"\midrule"]
        for name, keys, tr, d in metrics:
            vals = [ncu_val(N["kernels"][k], *keys) for k in kk]
            if all(v is None for v in vals):
                continue
            L.append(name + " & " + " & ".join("--" if v is None else f((tr(v) if tr else v), d) for v in vals) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # ---- block sweep figure ----
    B = K.get("medium", {}).get("blocks")
    if B:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3.3, 2.1))
        for kn, st in (("gather", "o-"), ("gather_xr", "s--"), ("gather_fm", "^-"), ("gather_cs", "d:")):
            pts = [(int(b), v) for b, v in B.get(kn, {}).items() if isinstance(v, float)]
            if pts:
                xs, ys = zip(*sorted(pts))
                ax.plot(xs, ys, st, label=kn, ms=4, lw=1.2)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("threads per block")
        ax.set_ylabel("kernel ms (medium)")
        ax.legend(fontsize=6, frameon=False)
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        FIG.mkdir(exist_ok=True)
        fig.savefig(FIG / "ir2_blocks.pdf")
        L += [r"\begin{figure}[t]", r"\centering", r"\includegraphics[width=\columnwidth]{figures/ir2_blocks.pdf}",
              r"\caption{Block-size sweep on the medium instance (median of 10). gather\_cs stops at 256 threads,"
              r" where its shared memory reaches 33 KB per block.}", r"\label{fig:blocks}", r"\end{figure}", ""]

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("% generated by report/ir2_tables.py, do not edit\n" + "\n".join(L) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
