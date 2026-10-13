"""Figure 2: time of each implementation per problem size, on a log axis (dot plot).

Called by report/fill_results.py with either the Oct 6 numbers or a notebook results file.
A dot is a median. A horizontal bar through it spans the fastest and slowest sample, when samples exist.
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from style import CPU, GATHER, GRID, INK, MUTED, SCATTER, plt  # noqa: E402

LOOK = {   # label, colour, marker
    "cpu_loop": ("CPU loop (mirrors R)", CPU, "s"),
    "cpu_batched": ("CPU scipy product", CPU, "D"),
    "cpu_cpp": ("CPU C++ (Alg. 1)", CPU, "^"),
    "scatter": ("CUDA scatter", SCATTER, "o"),
    "gather": ("CUDA gather", GATHER, "o"),
}
ORDER = ["cpu_loop", "cpu_batched", "cpu_cpp", "scatter", "gather"]


def draw(panels, out):
    """panels: list of (title, {method: (median_ms, lo_ms, hi_ms, label)}), lo and hi may be None."""
    methods = [m for m in ORDER if any(m in p[1] for p in panels)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(3.35, 0.36 + 0.92 * len(panels) + 0.02 * len(methods)),
                             sharex=True)
    axes = [axes] if len(panels) == 1 else list(axes)
    values = [v[0] for p in panels for v in p[1].values()]
    lo, hi = min(values) / 2.2, max(values) * 9
    for ax, (title, data) in zip(axes, panels):
        for i, m in enumerate(methods):
            if m not in data:
                ax.text(lo * 1.6, i, "not run", va="center", ha="left", fontsize=6.2, color=MUTED)
                continue
            med, a, b, label = data[m]
            name, colour, marker = LOOK[m]
            if a is not None and b is not None and b > a:
                ax.plot([a, b], [i, i], color=colour, lw=1.6, solid_capstyle="butt", zorder=2)
            ax.plot([med], [i], marker=marker, ms=4.6, color=colour, mec="white", mew=0.6, zorder=3)
            ax.text(med * 1.35, i, label, va="center", ha="left", fontsize=6.2, color=INK)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([LOOK[m][0] for m in methods])
        ax.set_ylim(len(methods) - 0.45, -0.55)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.set_title(title, loc="left", fontsize=7, color=INK, pad=3)
    axes[-1].set_xlabel("time in ms, log scale (lower is better)")
    fig.subplots_adjust(hspace=0.42)
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=220)
    plt.close(fig)
    return out
