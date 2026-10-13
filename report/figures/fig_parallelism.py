"""Figure 1: how each kernel maps threads onto the data. Drawn from a real toy instance, so the
picture matches the code: 4 genes, 8 cells, 2 donors, K = 3, and warps drawn with 4 lanes instead of 32.

    python report/figures/fig_parallelism.py        # writes report/figures/fig_parallelism.pdf
"""
import pathlib
import sys

import numpy as np
import scipy.sparse as sp

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from pbulk.layout import build_donor_layout  # noqa: E402
from pbulk.reference import make_selection  # noqa: E402
from style import GATHER, INK, MUTED, SCATTER, TINT_A, TINT_B, plt  # noqa: E402

G, C, D, K, LANES = 4, 8, 2, 3, 4


def toy():
    rng = np.random.default_rng(5)
    nz = rng.random((G, C)) < 0.7
    nz[G - 1, :4] = True                               # the last gene is expressed in all of donor A, for the example
    X = sp.csc_matrix(np.where(nz, 1.0, 0.0).astype(np.float32))
    subject = np.array(["A"] * 4 + ["B"] * 4)
    S, _, _ = make_selection(subject, K=K, ratio_min=0.5, ratio_max=0.9, seed=3)
    return X, S, subject


def box(ax, x, y, w, h, fc="white", ec=INK, lw=0.5, z=2, r=0.0):
    if r:
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z)
    else:
        p = Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(p)


def arrow(ax, a, b, color=INK, lw=0.6, ls="-", rad=0.0, z=3):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=5, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0))


def txt(ax, x, y, s, size=6, color=INK, ha="center", va="center", weight="normal", style="normal"):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, weight=weight, style=style, zorder=5)


def setup(ax, title):
    ax.set_xlim(0, 17)
    ax.set_ylim(15.6, -0.9)
    ax.axis("off")
    ax.set_aspect("equal")
    txt(ax, 0, -0.4, title, size=7.5, ha="left", weight="bold")


def draw_scatter(ax, X, S):
    setup(ax, "(a) Scatter: a block per pseudo-bulk, a warp per cell")
    cx = lambda c: 1.7 + 1.95 * c                     # centre of cell c's column
    txt(ax, 0, 0.5, "X stored by cell (CSC): each cell lists its nonzero genes", ha="left", color=MUTED)
    for d, tint in enumerate((TINT_A, TINT_B)):
        box(ax, cx(4 * d) - 0.9, 1.0, 4 * 1.95 - 0.15, 3.9, fc=tint, ec="none", z=0)
        txt(ax, cx(4 * d + 1.5), 1.35, f"donor {'AB'[d]}", color=MUTED)
    sel = [list(S.indices[S.indptr[p]:S.indptr[p + 1]]) for p in range(2)]    # pseudo-bulks 0 and 1 (donor A)
    for c in range(C):
        genes = X.indices[X.indptr[c]:X.indptr[c + 1]]
        both = c in sel[0] and c in sel[1]
        for i, g in enumerate(genes):
            box(ax, cx(c) - 0.42, 1.75 + 0.62 * i, 0.84, 0.52, fc="white", ec=SCATTER if both else INK,
                lw=1.0 if both else 0.5)
            txt(ax, cx(c), 2.01 + 0.62 * i, f"g{g}", size=5.5)
        txt(ax, cx(c), 4.6, f"cell {c}", size=5.5, color=SCATTER if both else MUTED)

    txt(ax, 16.9, 5.25, "a cell chosen by t pseudo-bulks", size=5.6, color=SCATTER, ha="right")
    txt(ax, 16.9, 5.8, "is read t times (orange: both blocks)", size=5.6, color=SCATTER, ha="right")
    bx, bw, ww, top = [0.3, 8.8], 7.9, 3.65, 6.3
    for p in range(2):
        box(ax, bx[p], top, bw, 3.1, fc="white", ec=INK, lw=0.7, r=0.15)
        ax.text(bx[p] + 0.25, top + 0.4, f"block {p} = pseudo-bulk {p}", ha="left", va="center", weight="bold",
                fontsize=6, zorder=6, bbox=dict(fc="white", ec="none", pad=0.6))
        px = bx[p] + 1.0
        for g in range(G):
            box(ax, px + 1.5 * g, 11.2, 1.4, 0.8, fc="white", ec=INK)
            txt(ax, px + 1.5 * g + 0.7, 11.6, f"g{g}", size=5.5)
        txt(ax, bx[p] + bw / 2, 12.4, f"PB row {p}", size=5.8, color=MUTED)
        hits = {}
        for w in range(2):                             # warp w takes selected cells w, w + 2, ...
            wx = bx[p] + 0.2 + (ww + 0.2) * w
            box(ax, wx, top + 0.8, ww, 2.05, fc="#fff4ee", ec=SCATTER, lw=0.6, r=0.1)
            mine = sel[p][w::2]
            txt(ax, wx + ww / 2, top + 1.2, f"warp {w}: cell " + ", ".join(map(str, mine)), size=5.6)
            for l in range(LANES):
                box(ax, wx + 0.2 + 0.83 * l, top + 1.65, 0.75, 0.75, fc="white", ec=SCATTER, lw=0.5)
                txt(ax, wx + 0.575 + 0.83 * l, top + 2.03, f"L{l}", size=5)
            for c in mine:
                arrow(ax, (cx(c), 4.9), (wx + ww / 2, top + 0.78),
                      color=SCATTER if c in sel[1 - p] else MUTED, lw=0.8)
                for l, g in enumerate(X.indices[X.indptr[c]:X.indptr[c + 1]][:LANES]):   # lane l takes nonzero l
                    hits[g] = hits.get(g, 0) + 1
                    arrow(ax, (wx + 0.575 + 0.83 * l, top + 2.42), (px + 1.5 * g + 0.4 + 0.3 * (hits[g] - 1), 11.18),
                          color=SCATTER, lw=0.5)
    txt(ax, 8.5, 13.3, "lanes of different warps add into the same entry,", size=5.6, color=SCATTER)
    txt(ax, 8.5, 13.85, "so every add is atomicAdd(&PB[p][g], v)", size=5.6, color=SCATTER)
    txt(ax, 8.5, 14.7, "Row p is written by block p only. Inside a block the", size=5.6, color=MUTED)
    txt(ax, 8.5, 15.25, "writes collide, and the atomics serialise them.", size=5.6, color=MUTED)


def draw_gather(ax, X, S, subject):
    setup(ax, "(b) Gather: a warp per (gene, donor) segment")
    lay = build_donor_layout(X, S, subject, K)
    seg, col, mask = lay["seg_ptr"], lay["col"], lay["mask"]
    txt(ax, 0, 0.5, "X stored by gene (CSR), cells sorted by donor", ha="left", color=MUTED)
    x0, w, rh = 2.1, 0.95, 0.85
    g, d = G - 1, 0                                    # the highlighted segment: last gene, donor A
    src = {}
    for gg in range(G):
        txt(ax, x0 - 0.3, 1.9 + rh * gg, f"gene {gg}", size=5.5, ha="right", color=MUTED)
        x = x0
        for dd in range(D):
            lo, hi = seg[gg * D + dd], seg[gg * D + dd + 1]
            on = (gg, dd) == (g, d)
            box(ax, x - 0.1, 1.52 + rh * gg, 4 * w + 0.1, rh - 0.1, fc=(TINT_A, TINT_B)[dd],
                ec=GATHER if on else "none", lw=1.1, z=1, r=0.08)
            for i in range(lo, hi):
                box(ax, x + (i - lo) * w, 1.62 + rh * gg, w - 0.12, rh - 0.3, fc="white",
                    ec=GATHER if on else INK, lw=0.9 if on else 0.5)
                txt(ax, x + (i - lo) * w + (w - 0.12) / 2, 1.9 + rh * gg, f"c{col[i]}", size=5.3)
                if on:
                    src[i - lo] = x + (i - lo) * w + (w - 0.12) / 2
            x += 4 * w + 0.35
    txt(ax, x0 + 2 * w, 1.2, "donor A cells", size=5.2, color=MUTED)
    txt(ax, x0 + 6 * w + 0.35, 1.2, "donor B cells", size=5.2, color=MUTED)

    mx = 11.4
    txt(ax, mx + 2.7, 0.5, "mask[c]: bit k = cell c is in", size=5.4, color=MUTED)
    txt(ax, mx + 2.7, 1.0, "pseudo-bulk k of its donor", size=5.4, color=MUTED)
    for c in range(C):
        bits = "".join(str((int(mask[c]) >> k) & 1) for k in range(K))
        r, cc = divmod(c, 4)
        box(ax, mx + 2.8 * r, 1.6 + 0.72 * cc, 2.6, 0.6, fc=(TINT_A, TINT_B)[r], ec=INK)
        txt(ax, mx + 2.8 * r + 1.3, 1.9 + 0.72 * cc, f"c{c}: {bits}", size=5.3)

    lo, hi = seg[g * D + d], seg[g * D + d + 1]
    wx0, wy, lw_ = 0.3, 6.1, 3.85
    box(ax, wx0, wy, 16.4, 3.9, fc="white", ec=GATHER, lw=0.8, r=0.15)
    txt(ax, wx0 + 0.25, wy + 2.98, f"one warp owns the segment (gene {g}, donor A)", ha="left", weight="bold", size=6)
    txt(ax, wx0 + 16.15, wy + 2.98, "4 lanes drawn, 32 in CUDA", ha="right", size=5.4, color=MUTED)
    for l in range(LANES):
        lx = wx0 + 0.25 + (lw_ + 0.17) * l
        box(ax, lx, wy + 0.35, lw_, 2.2, fc="#eef4fc", ec=GATHER, lw=0.5, r=0.08)
        if lo + l < hi:
            c = col[lo + l]
            bits = [(int(mask[c]) >> k) & 1 for k in range(K)]
            txt(ax, lx + lw_ / 2, wy + 0.72, f"lane {l}: c{c}, mask " + "".join(map(str, bits)), size=5.4)
            arrow(ax, (src[l], 1.62 + rh * g + rh - 0.28), (lx + lw_ / 2, wy + 0.33), color=GATHER, lw=0.7)
        else:
            bits = [0] * K
            txt(ax, lx + lw_ / 2, wy + 0.72, f"lane {l}: nothing left", size=5.4, color=MUTED)
        for k in range(K):
            box(ax, lx + 0.2 + 1.18 * k, wy + 1.12, 1.08, 0.72, fc="white" if bits[k] else "#e9e8e3", ec=GATHER, lw=0.5)
            txt(ax, lx + 0.74 + 1.18 * k, wy + 1.48, "+ v" if bits[k] else "+ 0", size=5.4,
                color=INK if bits[k] else MUTED)
        txt(ax, lx + lw_ / 2, wy + 2.2, "acc[0]  acc[1]  acc[2]", size=5, color=MUTED)
    txt(ax, wx0 + 8.2, wy + 3.45, "each nonzero is read once and added to K accumulators held in registers",
        size=5.6, color=GATHER)

    txt(ax, 8.5, 10.7, "warp shuffles sum the accumulators across lanes in lg 32 = 5 steps,",
        size=5.6, color=GATHER)
    txt(ax, 8.5, 11.25, "then lane 0 stores the K results", size=5.6, color=GATHER)
    px, py, ph = 3.7, 12.0, 0.56
    for p in range(D * K):
        for gg in range(G):
            on = gg == g and p // K == d
            box(ax, px + 1.5 * gg, py + ph * p, 1.4, ph - 0.1, fc="#d9e7f8" if on else "white",
                ec=GATHER if on else MUTED, lw=0.9 if on else 0.35)
        txt(ax, px - 0.2, py + ph * p + 0.23, f"PB row {p} ({'AB'[p // K]}, k={p % K})", size=4.9, ha="right",
            color=MUTED)
    for gg in range(G):
        txt(ax, px + 1.5 * gg + 0.7, py - 0.3, f"g{gg}", size=5, color=MUTED)
    for k in range(K):
        arrow(ax, (px + 1.5 * g + 3.0, py + ph * k + 0.23), (px + 1.5 * g + 1.45, py + ph * k + 0.23),
              color=GATHER, lw=0.6)
    txt(ax, px + 1.5 * g + 3.15, py + 0.2, "K plain stores", size=5.6, ha="left", color=GATHER)
    txt(ax, px + 1.5 * g + 3.15, py + 1.35, "each entry has", size=5.6, ha="left", color=MUTED)
    txt(ax, px + 1.5 * g + 3.15, py + 1.9, "exactly one", size=5.6, ha="left", color=MUTED)
    txt(ax, px + 1.5 * g + 3.15, py + 2.45, "writer: no race", size=5.6, ha="left", color=MUTED)


def main(out=HERE / "fig_parallelism.pdf"):
    X, S, subject = toy()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.45))
    draw_scatter(axes[0], X, S)
    draw_gather(axes[1], X, S, subject)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.04)
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=220)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(main())
