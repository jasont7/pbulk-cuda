"""Experiments for Interim Report 1. Writes one results JSON that report/build.py turns into tables and plots.

    python -m bench.ir1 --out results/ir1_results.json              # on a GPU (Colab T4)
    python -m bench.ir1 --backend emulator --tiny --out /tmp/x.json   # local dry run, no GPU, timings meaningless

Stages:
  correctness  both kernels against a float64 reference on instances that need one warp, one block of
               several warps, and several blocks, plus edge cases and a random sweep
  kernel       kernel time without transfers (CUDA events) against the CPU baselines, per size.
               Scatter against gather is the layout ablation. Several trials, all samples kept.
  end_to_end   wall time including host preparation, host-to-device copies, launch and copy back (medium size)
  profile      nvidia-smi sampling, ptxas register usage, Nsight Compute or nvprof if present
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # the CPU baselines are single-threaded

import argparse
import datetime as dt
import gc
import hashlib
import json
import pathlib
import platform
import time
import traceback

import numpy as np
import scipy
import scipy.sparse as sp

from bench import backends, profiling
from pbulk import cpu_cpp
from pbulk.launch import WARP, gather_launch, regime, scatter_launch
from pbulk.layout import build_donor_layout, gather_arrays, scatter_arrays
from pbulk.reference import compute_batched, compute_loop, make_selection
from pbulk.synthetic import make_synthetic

SCHEMA = 1
ROOT = pathlib.Path(__file__).resolve().parent.parent

# Synthetic problems (pbulk.synthetic, 8% dense, CPM columns). Medium and large are the two sizes of RESULTS.md.
SIZES = {
    "small": dict(genes=2000, cells=20000, donors=30, K=30),
    "medium": dict(genes=4000, cells=50000, donors=30, K=30),
    "large": dict(genes=8000, cells=120000, donors=30, K=30),
}
TINY_SIZES = {   # for the emulator dry run only
    "small": dict(genes=6, cells=300, donors=2, K=3),
    "medium": dict(genes=8, cells=400, donors=2, K=4),
    "large": dict(genes=10, cells=500, donors=3, K=4),
}

# Hand-picked correctness instances. sb and gb are the block sizes of the scatter and gather launches.
# ratio is the range of the fraction of a donor's cells that each pseudo-bulk samples.
CASES = [
    dict(name="one warp (gather)", genes=1, cells=200, donors=1, K=30, density=0.6, sb=32, gb=32),
    dict(name="one warp (scatter)", genes=24, cells=60, donors=1, K=1, density=0.3, sb=32, gb=1024),
    dict(name="one block, several warps", genes=32, cells=300, donors=1, K=1, density=0.3, sb=256, gb=1024),
    dict(name="several blocks, last one partial", genes=7, cells=500, donors=3, K=5, density=0.2, sb=256, gb=128),
    dict(name="K = 32, every mask bit used", genes=6, cells=400, donors=2, K=32, density=0.3, sb=256, gb=128),
    dict(name="every cell selected", genes=6, cells=300, donors=2, K=30, density=0.3, sb=256, gb=128,
         ratio=(1.0, 1.0)),
    dict(name="very sparse, empty segments", genes=16, cells=400, donors=3, K=4, density=0.01, sb=64, gb=128),
    dict(name="dense, segments of exactly 64", genes=5, cells=128, donors=2, K=3, density=1.0, sb=256, gb=64),
    dict(name="dense, ragged segments", genes=5, cells=131, donors=2, K=3, density=1.0, sb=256, gb=64),
    dict(name="a donor with one cell", genes=6, cells=200, donors=3, K=4, density=0.3, sb=128, gb=128,
         lone_donor=True),
    dict(name="long segments", genes=2, cells=4000, donors=2, K=8, density=0.9, sb=512, gb=32),
]


def now():
    return time.perf_counter()


def make_case(genes, cells, donors, K, density, seed=0, integer=False, ratio=(0.3, 1.0), lone_donor=False, **_):
    """A small random instance. Integer values make every float32 sum exact, whatever the order."""
    rng = np.random.default_rng(seed)
    nz = rng.random((genes, cells)) < density
    vals = rng.integers(1, 8, size=(genes, cells)) if integer else rng.lognormal(size=(genes, cells))
    X = sp.csc_matrix(np.where(nz, vals, 0).astype(np.float32))
    donor = rng.permutation(np.arange(cells) % donors)       # every donor gets at least one cell
    if lone_donor and donors > 1:
        extra = np.flatnonzero(donor == donors - 1)[1:]
        donor[extra] = 0                                      # the last donor keeps a single cell
    subject = np.array([f"donor{d:02d}" for d in donor])
    S, _, _ = make_selection(subject, K=K, ratio_min=ratio[0], ratio_max=ratio[1], seed=seed + 1)
    return X, S, subject


def random_cases(n, seed=2026, big=False):
    """Random shapes and block sizes. `big` uses shapes that only make sense on a real device."""
    rng = np.random.default_rng(seed)
    blocks = [32, 64, 128, 256, 512, 1024]
    out = []
    for i in range(n):
        donors = int(rng.integers(1, 7))
        genes = int(rng.integers(1, 2000 if big else 12))
        out.append(dict(name=f"random {i}", genes=genes, donors=donors, K=int(rng.integers(1, 33)),
                        cells=int(rng.integers(donors, 3000 if big else 400)),
                        density=float(rng.uniform(0.02, 0.9)),
                        sb=int(rng.choice(blocks)), gb=int(rng.choice(blocks)), seed=int(rng.integers(1 << 30))))
    return out


def reference64(X, S):
    """Unnormalised PB = S X^T in float64."""
    return (S.astype(np.float64) @ X.T.tocsr().astype(np.float64)).toarray()


def compare(out, ref, integer):
    """Error of an unnormalised kernel output against the float64 reference."""
    out = np.asarray(out, dtype=np.float64)
    nonzero = ref != 0
    rel = float(np.max(np.abs(out - ref)[nonzero] / np.abs(ref)[nonzero])) if nonzero.any() else 0.0
    zeros_ok = bool(np.all(out[~nonzero] == 0))
    exact = bool(np.array_equal(out, ref))
    ok = exact if integer else (zeros_ok and bool(np.allclose(out, ref, rtol=1e-4, atol=0)))
    return dict(max_rel_err=rel, exact=exact, ok=bool(ok))


def run_case(backend, case, integer):
    X, S, subject = make_case(**{**case, "seed": case.get("seed", 0)}, integer=integer)
    G, K = X.shape[0], case["K"]
    n_pb, D = S.shape[0], S.shape[0] // K
    ref = reference64(X, S)
    res = dict(name=case["name"], integer=integer, nnz=int(X.nnz),
               **{k: case[k] for k in ("genes", "cells", "donors", "K", "density")})
    dev = backend.to_device(scatter_arrays(X, S))
    launch = scatter_launch(n_pb, case["sb"])
    out = backend.to_host(backend.scatter_kernel(dev, n_pb, G, case["sb"], normalise=False))
    res["scatter"] = dict(**launch, regime=regime(launch), **compare(out, ref, integer))
    dev = backend.to_device(gather_arrays(build_donor_layout(X, S, subject, K)))
    launch = gather_launch(G, D, case["gb"])
    out = backend.to_host(backend.gather_kernel(dev, G, D, K, case["gb"], normalise=False))
    res["gather"] = dict(**launch, regime=regime(launch), **compare(out, ref, integer))
    return res


def correctness(backend, n_random=25, big=True):
    cases = []
    for case in CASES + random_cases(n_random, big=big):
        for integer in (False, True):
            cases.append(run_case(backend, case, integer))
    regimes = {k: sorted({c[k]["regime"] for c in cases if c[k]["ok"]}) for k in ("scatter", "gather")}
    return dict(cases=cases, n_cases=len(cases), regimes_passed=regimes,
                all_ok=all(c[k]["ok"] for c in cases for k in ("scatter", "gather")))


def make_problem(cfg):
    X, meta = make_synthetic(cfg["genes"], cfg["cells"], cfg["donors"])
    subject = meta["SubjectName"].values
    S, _, _ = make_selection(subject, K=cfg["K"])
    return X, S, subject, cfg["K"]


def work_counts(X, S, K, n_donors):
    """Operation counts of each algorithm on this instance. Counted, not measured."""
    nnz_per_cell = np.diff(X.tocsc().indptr)
    G, n_pb = X.shape[0], S.shape[0]
    terms = int(nnz_per_cell[S.indices].sum())            # F: one addition per term of S X^T
    return dict(nnz=int(X.nnz), selections=int(S.nnz), terms=terms,
                mean_selections_per_cell=float(S.nnz / X.shape[1]),
                scatter_atomic_adds=terms, gather_masked_adds=int(WARP * X.nnz),
                gather_nonzero_reads=int(X.nnz), scatter_nonzero_reads=terms,
                output_entries=int(n_pb * G), segments=int(G * n_donors),
                scatter_launch=scatter_launch(n_pb), gather_launch=gather_launch(G, n_donors))


def cpu_samples(fn, repeats):
    out, times = None, []
    for _ in range(repeats):
        t = now()
        out = fn()
        times.append((now() - t) * 1e3)
    return out, times


def rel_err(out, ref):
    """Same error measure as RESULTS.md: normalised output against the normalised CPU reference."""
    return float(np.max(np.abs(out - ref) / (np.abs(ref) + 1e-3)))


def end_to_end(backend, X, S, subject, K, repeats=5):
    """Wall time from host arrays to the result back on the host, split into stages (ms).

    One untimed warm-up of each path comes first, so kernel compilation and the first device
    allocations are not counted. The cell selection S is an input to every path, CPU and GPU alike.
    """
    G, n_pb = X.shape[0], S.shape[0]
    D = n_pb // K

    def scatter_once():
        t0 = now(); arrays = scatter_arrays(X, S)
        t1 = now(); dev = backend.to_device(arrays)
        t2 = now(); PB = backend.scatter_kernel(dev, n_pb, G); backend.sync()
        t3 = now(); out = backend.to_host(PB)
        t4 = now()
        return out, arrays, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)

    def gather_once():
        t0 = now(); arrays = gather_arrays(build_donor_layout(X, S, subject, K))
        t1 = now(); dev = backend.to_device(arrays)
        t2 = now(); PB = backend.gather_kernel(dev, G, D, K); backend.sync()
        t3 = now(); out = backend.to_host(PB)
        t4 = now()
        return out, arrays, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)

    res = {"repeats": repeats, "stages": ["host_prep_ms", "h2d_ms", "kernel_ms", "d2h_ms"],
           "note": "host_prep is the dtype conversion for scatter and the layout build for gather. "
                   "kernel is launch + kernel + row normalisation, synchronised. Pageable host memory."}
    for name, once in (("scatter", scatter_once), ("gather", gather_once)):
        out, arrays, _ = once()
        rows = [[s * 1e3 for s in once()[2]] for _ in range(repeats)]
        entry = {stage: [r[i] for r in rows] for i, stage in enumerate(res["stages"])}
        entry["total_ms"] = [sum(r) for r in rows]
        entry["total_without_host_prep_ms"] = [sum(r[1:]) for r in rows]
        entry["bytes_h2d"] = int(sum(a.nbytes for a in arrays))
        entry["bytes_d2h"] = int(np.asarray(out).nbytes)
        res[name] = entry
        del out, arrays
        backend.free()
    return res


def profile_stage(backend, run_scatter, run_gather, size, seconds):
    """Utilisation while each kernel runs in a loop, plus the static and counter-based tools."""
    def loop(fn):
        def call():
            fn()
            backend.sync()
        return call
    out = {"size": size, "smi": {}}
    for name, fn in (("scatter", run_scatter), ("gather", run_gather)):
        out["smi"][name] = profiling.smi_sample(loop(fn), seconds=seconds)
    out["resource_usage"] = profiling.resource_usage()
    return out


def run_size(backend, name, cfg, trials=3, repeats=10, cpu_repeats=3, loop=False, e2e=False,
             profile=False, profile_seconds=6.0):
    """Everything measured on one problem size. Returns {"kernel": ..., "end_to_end": ..., "profile": ...}."""
    X, S, subject, K = make_problem(cfg)
    G, n_pb = X.shape[0], S.shape[0]
    D = n_pb // K
    res = dict(size=name, **cfg, n_pb=n_pb, work=work_counts(X, S, K, D), trials=trials, repeats=repeats)
    print(f"[{name}] X {X.shape} nnz={X.nnz:,}  S {S.shape} nnz={S.nnz:,}", flush=True)

    ref, res["cpu_batched_ms"] = cpu_samples(lambda: compute_batched(X, S), cpu_repeats)
    if cpu_cpp.available():
        args = cpu_cpp.prepare(X, S)
        out, res["cpu_cpp_ms"] = cpu_samples(lambda: cpu_cpp.compute(args), cpu_repeats)
        res["cpu_cpp_max_rel_err"] = rel_err(out.astype(np.float64), ref)
        del args, out
    if loop:
        _, res["cpu_loop_ms"] = cpu_samples(lambda: compute_loop(X, S), 1)

    t = now(); lay = build_donor_layout(X, S, subject, K); res["layout_build_ms"] = (now() - t) * 1e3
    t = now(); s_dev = backend.to_device(scatter_arrays(X, S)); res["scatter_upload_ms"] = (now() - t) * 1e3
    t = now(); g_dev = backend.to_device(gather_arrays(lay)); res["gather_upload_ms"] = (now() - t) * 1e3
    del lay
    run = {"scatter": lambda: backend.scatter_kernel(s_dev, n_pb, G),
           "gather": lambda: backend.gather_kernel(g_dev, G, D, K)}
    for k, fn in run.items():
        out = backend.to_host(fn()).astype(np.float64)
        res[f"{k}_max_rel_err"] = rel_err(out, ref)
        res[f"{k}_ok"] = bool(np.allclose(out, ref, rtol=1e-3, atol=1e-2))
        res[f"{k}_ms"] = []
        del out
    for trial in range(trials):                      # interleave the kernels, so drift hits both alike
        for k, fn in run.items():
            res[f"{k}_ms"].append(backend.time_samples(fn, repeats=repeats, warmup=2))
    for k in run:
        print(f"[{name}] {k:8s} median {np.median(res[k + '_ms']):10.3f} ms   max rel err "
              f"{res[k + '_max_rel_err']:.2e}   {'OK' if res[k + '_ok'] else 'FAIL'}", flush=True)

    out = {"kernel": res}
    if profile:
        out["profile"] = profile_stage(backend, run["scatter"], run["gather"], name, profile_seconds)
    del s_dev, g_dev, run
    backend.free()
    if e2e:
        e = end_to_end(backend, X, S, subject, K)
        e.update(size=name, cpu_batched_ms=res["cpu_batched_ms"], cpu_cpp_ms=res.get("cpu_cpp_ms"))
        out["end_to_end"] = e
    del X, S
    gc.collect()
    backend.free()
    return out


def environment(backend):
    cpu = platform.processor()
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    sha = hashlib.sha256((ROOT / "pbulk" / "kernels.cu").read_bytes()).hexdigest()
    return dict(backend=backend.name, data="synthetic (pbulk.synthetic)",
                created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                gpu=backend.device_info(), cpu=cpu, cpu_count=os.cpu_count(), platform=platform.platform(),
                python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                threads_env={v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                            "MKL_NUM_THREADS")},
                cpp_flags=cpu_cpp.FLAGS if cpu_cpp.available() else None, kernels_sha256=sha)


def med(x):
    return float(np.median(np.asarray(x, dtype=np.float64)))


def summary(r):
    """The results as a plain-text table."""
    L = [f"backend: {r['meta']['backend']}   device: {r['meta']['gpu'].get('name')}   data: {r['meta']['data']}"]
    if r["meta"]["backend"] != "cuda":
        L.append("NOT A GPU RUN. Timings below are meaningless and must not go in the report.")
    c = r.get("correctness")
    if c:
        L.append(f"\nCorrectness: {c['n_cases']} instances, all ok = {c['all_ok']}")
        for k in ("scatter", "gather"):
            L.append(f"  {k:8s} passed in regimes: {', '.join(c['regimes_passed'][k])}")
            worst = max(x[k]["max_rel_err"] for x in c["cases"])
            L.append(f"  {k:8s} worst relative error vs float64: {worst:.2e}")
    if r.get("kernel"):
        L.append("\nKernel time without transfers, median ms (CPU: median of its repeats)")
        L.append(f"  {'size':8s}{'nnz':>12s}{'CPU scipy':>12s}{'CPU C++':>12s}{'scatter':>12s}{'gather':>12s}"
                 f"{'g vs s':>9s}{'g vs C++':>10s}")
        for name, k in r["kernel"].items():
            s, g = med(k["scatter_ms"]), med(k["gather_ms"])
            cpp = med(k["cpu_cpp_ms"]) if k.get("cpu_cpp_ms") else float("nan")
            L.append(f"  {name:8s}{k['work']['nnz']:12,d}{med(k['cpu_batched_ms']):12.1f}{cpp:12.1f}"
                     f"{s:12.3f}{g:12.3f}{s / g:8.1f}x{cpp / g:9.1f}x")
    e = r.get("end_to_end")
    if e:
        L.append(f"\nEnd to end on '{e['size']}', median ms: host prep + H2D + kernel + D2H = total")
        for k in ("scatter", "gather"):
            p = [med(e[k][s]) for s in e["stages"]]
            L.append(f"  {k:8s} {p[0]:9.1f} + {p[1]:8.1f} + {p[2]:8.2f} + {p[3]:7.2f} = {med(e[k]['total_ms']):9.1f}")
        L.append(f"  CPU scipy {med(e['cpu_batched_ms']):9.1f}"
                 + (f"   CPU C++ {med(e['cpu_cpp_ms']):9.1f}" if e.get("cpu_cpp_ms") else ""))
    p = r.get("profile")
    if p:
        L.append(f"\nProfile on '{p['size']}'")
        for k, s in p["smi"].items():
            if s.get("available"):
                L.append(f"  nvidia-smi {k:8s} GPU util median {s['utilization.gpu']['median']:.0f}%   memory util "
                         f"median {s['utilization.memory']['median']:.0f}%   ({s['samples']} samples)")
            else:
                L.append(f"  nvidia-smi {k:8s} unavailable: {s.get('reason')}")
        for k, v in p.get("resource_usage", {}).get("kernels", {}).items():
            L.append(f"  ptxas {k}: {v.get('registers')} registers, {v.get('stack_bytes')} bytes stack frame")
        for tool in ("ncu", "nvprof"):
            t = p.get(tool, {})
            L.append(f"  {tool}: {'collected' if t.get('available') else 'unavailable (' + str(t.get('reason')) + ')'}")
    for err in r.get("errors", []):
        L.append(f"\nERROR in {err['stage']}: {err['error']}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="cuda", choices=["cuda", "emulator"])
    ap.add_argument("--out", default="results/ir1_results.json")
    ap.add_argument("--tiny", action="store_true", help="tiny sizes, for the emulator dry run")
    ap.add_argument("--stages", default="correctness,kernel,profile_tools")
    ap.add_argument("--profile-seconds", type=float, default=6.0)
    a = ap.parse_args(argv)

    backend = backends.get(a.backend)
    sizes = TINY_SIZES if a.tiny else SIZES
    stages = set(a.stages.split(","))
    r = {"schema": SCHEMA, "meta": environment(backend), "errors": []}
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        out.write_text(json.dumps(r, indent=1, default=str))

    def stage(name, fn):
        print(f"== {name}", flush=True)
        try:
            fn()
        except Exception:  # noqa: BLE001  keep going, record what failed
            r["errors"].append({"stage": name, "error": traceback.format_exc(limit=6)})
            print(traceback.format_exc(limit=6), flush=True)
        save()

    if "correctness" in stages:
        def do():
            r["correctness"] = correctness(backend, n_random=3 if a.tiny else 25, big=not a.tiny)
        stage("correctness", do)

    if "kernel" in stages:
        r["kernel"] = {}
        for name, cfg in sizes.items():
            def do(name=name, cfg=cfg):
                res = run_size(backend, name, cfg, trials=1 if a.tiny else 3, repeats=2 if a.tiny else 10,
                               loop=(name == "medium"), e2e=(name == "medium"), profile=(name == "large"),
                               profile_seconds=a.profile_seconds)
                r["kernel"][name] = res["kernel"]
                for k in ("end_to_end", "profile"):
                    if k in res:
                        r[k] = res[k]
            stage(f"kernel timing, {name}", do)

    if "profile_tools" in stages and a.backend == "cuda":
        def do():
            r.setdefault("profile", {"size": "medium", "smi": {}})
            r["profile"]["ncu"] = profiling.ncu_profile("medium")
            r["profile"]["nvprof"] = profiling.nvprof_profile("medium")
        stage("command-line profilers", do)

    save()
    print("\n" + summary(r))
    print(f"\nwrote {out}")
    return r


if __name__ == "__main__":
    main()
