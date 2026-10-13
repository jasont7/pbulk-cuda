"""Time the CPU reference paths on synthetic data. The same harness will time the CUDA path."""
import argparse
import time

import numpy as np

from pbulk.reference import make_selection, compute_loop, compute_batched, fractions
from pbulk.synthetic import make_synthetic


def timed(fn, repeats):
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return out, best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", type=int, default=2000)
    ap.add_argument("--cells", type=int, default=20000)
    ap.add_argument("--donors", type=int, default=30)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()

    X, meta = make_synthetic(a.genes, a.cells, a.donors)
    S, _, _ = make_selection(meta["SubjectName"].values, K=a.K)
    print(f"X {X.shape} nnz={X.nnz:,}  S {S.shape} nnz={S.nnz:,}")

    loop, t_loop = timed(lambda: compute_loop(X, S), a.repeats)
    batch, t_batch = timed(lambda: compute_batched(X, S), a.repeats)
    _, t_frac = timed(lambda: [fractions(S, meta[c].values) for c in
                               ("cell_type1", "cell_type2", "cell_type3")], a.repeats)
    print(f"loop    {t_loop:8.3f} s")
    print(f"batched {t_batch:8.3f} s   ({t_loop / t_batch:.1f}x vs loop)")
    print(f"fractions {t_frac:6.3f} s")
    print(f"max rel diff loop vs batched: {np.max(np.abs(loop - batch) / (np.abs(loop) + 1e-9)):.2e}")
