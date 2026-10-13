"""Check both CUDA kernels against the CPU reference on the same selection, then time everything.

    python -m bench.run_gpu --cells 50000 --genes 4000
    python -m bench.run_gpu --data data/       # real data exported on Narval
"""
import argparse
import json
import pathlib
import time

import numpy as np

from pbulk.reference import make_selection, compute_loop, compute_batched
from pbulk.synthetic import make_synthetic


def load_real(path):
    import pandas as pd
    import scipy.sparse as sp
    X = sp.load_npz(pathlib.Path(path) / "cpm.npz").tocsc()
    meta = pd.read_csv(pathlib.Path(path) / "meta.csv")
    return X, meta


def cpu_ms(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return out, best * 1e3


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", type=int, default=4000)
    ap.add_argument("--cells", type=int, default=50000)
    ap.add_argument("--donors", type=int, default=30)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--data", default=None)
    ap.add_argument("--skip-loop", action="store_true", help="the CPU loop is slow on real data")
    ap.add_argument("--out", default="results/gpu.json")
    a = ap.parse_args()

    import cupy as cp
    from pbulk import gpu

    X, meta = load_real(a.data) if a.data else make_synthetic(a.genes, a.cells, a.donors)
    subject = meta["SubjectName"].values
    S, _, _ = make_selection(subject, K=a.K)
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"X {X.shape} nnz={X.nnz:,}   S {S.shape} nnz={S.nnz:,}")

    ref, t_batched = cpu_ms(lambda: compute_batched(X, S))
    res = {"genes": X.shape[0], "cells": X.shape[1], "nnz": int(X.nnz), "n_pb": S.shape[0],
           "cpu_batched_ms": t_batched}
    if not a.skip_loop:
        _, res["cpu_loop_ms"] = cpu_ms(lambda: compute_loop(X, S), repeats=1)

    t = time.perf_counter()
    d = gpu.GpuData(X, S, subject, a.K)
    cp.cuda.Device().synchronize()
    res["upload_and_layout_ms"] = (time.perf_counter() - t) * 1e3

    for name, fn in (("scatter", gpu.run_scatter), ("gather", gpu.run_gather)):
        out = cp.asnumpy(fn(d)).astype(np.float64)
        rel = np.max(np.abs(out - ref) / (np.abs(ref) + 1e-3))
        res[f"{name}_max_rel_err"] = float(rel)
        res[f"{name}_ms"] = gpu.time_gpu(lambda: fn(d))
        ok = "OK " if np.allclose(out, ref, rtol=1e-3, atol=1e-2) else "FAIL"
        print(f"{ok} {name:8s} {res[f'{name}_ms']:9.3f} ms   max rel err {rel:.2e}")

    for k in ("cpu_loop_ms", "cpu_batched_ms", "upload_and_layout_ms"):
        if k in res:
            print(f"{k:22s} {res[k]:10.1f} ms")
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
