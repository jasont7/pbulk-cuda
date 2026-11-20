"""Every IR2 experiment in one run, on a real GPU (Colab T4). Writes one JSON, saved after each stage.

    python -m bench.ir2 --out results/ir2_results.json

Kernels compared (pbulk/kernels.cu):
  scatter    IR1 v1, atomics
  gather     IR1 v2, the baseline for IR2
  gather_xr  idea 3a: transposed warp reduction only
  gather_fm  idea 3: fused mask stream + transposed reduction
  gather_cs  idea 2: complement + set-bit iteration in shared memory
  cusparse   library baseline, cuSPARSE SpGEMM through cupyx (S @ X^T)
Idea 1 (the layout built on the GPU) is measured in the layout and end-to-end stages.
"""
import argparse
import datetime as dt
import gc
import hashlib
import json
import pathlib
import subprocess
import sys
import time
import traceback

import numpy as np

from bench import profiling
from bench.ir1 import SIZES, TINY_SIZES, make_problem, reference64, rel_err, environment
from pbulk.layout import (build_donor_layout, complement_arrays, complement_work, donor_codes, fused_arrays,
                          gather_arrays)

ROOT = pathlib.Path(__file__).resolve().parent.parent
KERNELS = ("scatter", "gather", "gather_xr", "gather_fm", "gather_cs", "cusparse")
VARIANTS = ("gather_xr", "gather_fm", "gather_cs")


def now():
    return time.perf_counter()


def med(x):
    return float(np.median(np.ravel(x)))


def normalised64(PB):
    PB = np.asarray(PB, dtype=np.float64)
    return PB / PB.sum(axis=1, keepdims=True) * 1e6


def gpu_ms(gpu, fn, repeats=5):
    """Wall ms of fn() with a device sync, median of `repeats` after one warm-up. For multi-kernel host code."""
    fn(); gpu.sync()
    ts = []
    for _ in range(repeats):
        t = now(); fn(); gpu.sync(); ts.append((now() - t) * 1e3)
    return ts


def run_size(gpu, name, cfg, trials=3, repeats=10, blocks=False):
    import cupy as cp
    X, S, subject, K = make_problem(cfg)
    G, n_pb = X.shape[0], S.shape[0]
    D = n_pb // K
    res = dict(size=name, **cfg, n_pb=n_pb, nnz=int(X.nnz), selections=int(S.nnz), trials=trials, repeats=repeats)
    print(f"[{name}] X {X.shape} nnz={X.nnz:,}  S nnz={S.nnz:,}", flush=True)
    ref = normalised64(reference64(X, S))

    # ---- idea 1: the layout, host vs GPU ----
    t = now(); lay = build_donor_layout(X, S, subject, K); res["layout_host_ms"] = (now() - t) * 1e3
    res["work_cs"] = complement_work(lay)
    t = now(); codes, D2 = donor_codes(subject); res["codes_host_ms"] = (now() - t) * 1e3
    assert D2 == D
    t = now(); raw = gpu.GpuRaw(X, S, codes); res["raw_upload_ms"] = (now() - t) * 1e3
    res["raw_bytes"] = raw.nbytes()
    res["layout_gpu_ms"] = gpu_ms(gpu, lambda: gpu.layout_gpu(raw, D, K))
    lay_d = gpu.layout_gpu(raw, D, K)
    res["layout_identical"] = {k: bool(np.array_equal(np.asarray(lay[k]), cp.asnumpy(lay_d[k])))
                               for k in ("seg_ptr", "col", "val", "mask")}
    res["fm_prep_gpu_ms"] = gpu_ms(gpu, lambda: fused_arrays(lay_d))
    res["cs_prep_gpu_ms"] = gpu_ms(gpu, lambda: complement_arrays(lay_d, cp))
    print(f"[{name}] layout host {res['layout_host_ms']:.1f} ms, GPU {med(res['layout_gpu_ms']):.2f} ms, "
          f"identical {all(res['layout_identical'].values())}", flush=True)

    # ---- kernels ----
    base = gpu.to_device(gather_arrays(lay))
    fm = fused_arrays(lay_d)
    cs = complement_arrays(lay_d, cp)
    del raw
    from pbulk.layout import scatter_arrays
    sc = gpu.to_device(scatter_arrays(X, S))
    run = {"scatter": lambda: gpu.scatter_kernel(sc, n_pb, G),
           "gather": lambda: gpu.gather_kernel(base, G, D, K),
           "gather_xr": lambda: gpu.variant_kernel("gather_xr", base, G, D, K),
           "gather_fm": lambda: gpu.variant_kernel("gather_fm", fm, G, D, K),
           "gather_cs": lambda: gpu.variant_kernel("gather_cs", cs, G, D, K)}
    try:
        S32, XT = gpu.cusparse_operands(X, S)
        run["cusparse"] = lambda: gpu.cusparse_pb(S32, XT)
    except Exception as e:  # noqa: BLE001
        res["cusparse_error"] = repr(e)
    for k, fn in list(run.items()):
        try:
            out = normalised64(gpu.to_host(fn()))
            res[f"{k}_max_rel_err"] = rel_err(out, ref)
            res[f"{k}_ok"] = bool(np.allclose(out, ref, rtol=1e-3, atol=1e-2))
            res[f"{k}_ms"] = []
        except Exception as e:  # noqa: BLE001
            res[f"{k}_error"] = repr(e)
            run.pop(k)
    for _ in range(trials):
        for k, fn in run.items():
            res[f"{k}_ms"].append(gpu.time_samples(fn, repeats=repeats, warmup=2))
    for k in run:
        print(f"[{name}] {k:10s} median {med(res[k + '_ms']):9.3f} ms  err {res[k + '_max_rel_err']:.1e}  "
              f"{'OK' if res[k + '_ok'] else 'FAIL'}", flush=True)

    # ---- block-size sweep of the gather family ----
    if blocks:
        res["blocks"] = {}
        for k, arrs in (("gather", base), ("gather_xr", base), ("gather_fm", fm), ("gather_cs", cs)):
            res["blocks"][k] = {}
            for b in (32, 64, 128, 256, 512):
                if k == "gather_cs" and b * 33 * 4 > 48 * 1024:
                    continue
                try:
                    f = (lambda b=b, k=k, arrs=arrs: gpu.gather_kernel(arrs, G, D, K, block=b)) if k == "gather" else \
                        (lambda b=b, k=k, arrs=arrs: gpu.variant_kernel(k, arrs, G, D, K, block=b))
                    res["blocks"][k][str(b)] = med(gpu.time_samples(f, repeats=repeats, warmup=2))
                except Exception as e:  # noqa: BLE001
                    res["blocks"][k][str(b)] = repr(e)
    del base, fm, cs, sc, lay, lay_d, run
    gpu.free()
    return res, (X, S, subject, K)


def end_to_end(gpu, X, S, subject, K, repeats=5):
    """Host arrays in, normalised PB back on the host (ms per stage). One warm-up of each path first."""
    import cupy as cp
    G, n_pb = X.shape[0], S.shape[0]
    D = n_pb // K
    from pbulk.layout import scatter_arrays

    def scatter_once():
        t0 = now(); a = scatter_arrays(X, S)
        t1 = now(); d = gpu.to_device(a)
        t2 = now(); PB = gpu.scatter_kernel(d, n_pb, G); gpu.sync()
        t3 = now(); gpu.to_host(PB)
        t4 = now()
        return dict(host_prep_ms=t1 - t0, h2d_ms=t2 - t1, layout_gpu_ms=0.0, kernel_ms=t3 - t2, d2h_ms=t4 - t3)

    def ir1_gather_once():
        t0 = now(); a = gather_arrays(build_donor_layout(X, S, subject, K))
        t1 = now(); d = gpu.to_device(a)
        t2 = now(); PB = gpu.gather_kernel(d, G, D, K); gpu.sync()
        t3 = now(); gpu.to_host(PB)
        t4 = now()
        return dict(host_prep_ms=t1 - t0, h2d_ms=t2 - t1, layout_gpu_ms=0.0, kernel_ms=t3 - t2, d2h_ms=t4 - t3)

    def gpu_layout_once(which):
        def once():
            t0 = now(); codes, _ = donor_codes(subject)
            t1 = now(); raw = gpu.GpuRaw(X, S, codes)
            t2 = now(); lay = gpu.layout_gpu(raw, D, K)
            arrs = fused_arrays(lay) if which == "gather_fm" else gather_arrays(lay)
            del raw, lay
            gpu.sync()
            t3 = now()
            PB = gpu.variant_kernel(which, arrs, G, D, K) if which != "gather" else gpu.gather_kernel(arrs, G, D, K)
            gpu.sync()
            t4 = now(); gpu.to_host(PB)
            t5 = now()
            return dict(host_prep_ms=t1 - t0, h2d_ms=t2 - t1, layout_gpu_ms=t3 - t2, kernel_ms=t4 - t3, d2h_ms=t5 - t4)
        return once

    paths = {"scatter": scatter_once, "gather_ir1": ir1_gather_once,
             "gather_gpu_layout": gpu_layout_once("gather"), "gather_fm_gpu_layout": gpu_layout_once("gather_fm")}
    out = {"repeats": repeats, "note": "pageable host memory. layout_gpu includes the fused-mask prep for fm."}
    for name, once in paths.items():
        try:
            once(); gpu.free()
            rows = []
            for _ in range(repeats):
                rows.append({k: v * 1e3 for k, v in once().items()})
                gpu.free()
            e = {k: [r[k] for r in rows] for k in rows[0]}
            e["total_ms"] = [sum(r.values()) for r in rows]
            out[name] = e
            print(f"[e2e] {name:22s} total {med(e['total_ms']):8.1f} ms  "
                  + "  ".join(f"{k[:-3]} {med(v):.1f}" for k, v in e.items() if k != "total_ms"), flush=True)
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": repr(e), "trace": traceback.format_exc()[-1500:]}
    return out


NCU_SECTIONS = ("SpeedOfLight", "Occupancy", "LaunchStats", "WarpStateStats", "InstructionStats",
                "MemoryWorkloadAnalysis", "SchedulerStats")
NCU_METRICS = ("smsp__thread_inst_executed_per_inst_executed.ratio", "smsp__inst_executed.sum",
               "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum", "dram__bytes_read.sum",
               "l1tex__t_bytes.sum", "smsp__sass_inst_executed_op_shared_ld.sum",
               "smsp__sass_inst_executed_op_global_ld.sum")


def parse_ncu(text):
    import csv
    import io
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if "Metric Name" in l and "Metric Value" in l), None)
    if start is None:
        return {}
    out = {}
    for row in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        name, metric = (row.get("Kernel Name") or "").split("(")[0].strip(), row.get("Metric Name")
        if not name.startswith("pb_") or not metric:
            continue
        out.setdefault(name, {}).setdefault(metric, {"value": row.get("Metric Value"), "unit": row.get("Metric Unit")})
    return out


def ncu_stage(size="medium"):
    ncu = profiling._find("ncu", profiling.NCU_PATHS)
    log = []
    if ncu is None:
        log.append(profiling._try_install_ncu())
        ncu = profiling._find("ncu", profiling.NCU_PATHS)
    if ncu is None:
        return {"available": False, "log": log}
    cmd = [ncu, "--csv", "--kernel-name", "regex:^pb_", "--launch-count", "5"]
    for s in NCU_SECTIONS:
        cmd += ["--section", s]
    cmd += ["--metrics", ",".join(NCU_METRICS), sys.executable, "-m", "bench.ir2_profile_target", "--size", size]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=str(ROOT))
        k = parse_ncu(r.stdout)
        return {"available": bool(k), "tool": ncu, "size": size, "kernels": k, "exit": r.returncode,
                "log": log + [(r.stdout + r.stderr)[-2000:]] if not k else log}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "log": log + [repr(e)]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/ir2_results.json")
    ap.add_argument("--sizes", default="small,medium,large")
    ap.add_argument("--stages", default="kernel,e2e,ncu,resources")
    ap.add_argument("--tiny", action="store_true", help="tiny sizes, fake-cupy dry run only")
    ap.add_argument("--ncu-size", default="medium")
    ap.add_argument("--trials", type=int, default=3)
    a = ap.parse_args(argv)
    from pbulk import gpu
    from bench.backends import get
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    r = {"schema": "ir2-1", "meta": environment(get("cuda")), "errors": [], "kernel": {}, "end_to_end": {}}
    r["meta"]["kernels_sha256"] = hashlib.sha256((ROOT / "pbulk" / "kernels.cu").read_bytes()).hexdigest()

    def save():
        out.write_text(json.dumps(r, indent=1, default=float))

    stages = set(a.stages.split(","))
    problems = {}
    if "kernel" in stages:
        for name in a.sizes.split(","):
            try:
                r["kernel"][name], problems[name] = run_size(gpu, name, (TINY_SIZES if a.tiny else SIZES)[name],
                                                             trials=a.trials, blocks=(name == "medium"))
            except Exception:  # noqa: BLE001
                r["errors"].append({"stage": f"kernel/{name}", "trace": traceback.format_exc()[-3000:]})
                print(traceback.format_exc(), flush=True)
            save()
            gc.collect()
    if "e2e" in stages:
        for name in ("medium", "large"):
            try:
                X, S, subject, K = problems.get(name) or make_problem((TINY_SIZES if a.tiny else SIZES)[name])
                r["end_to_end"][name] = end_to_end(gpu, X, S, subject, K)
            except Exception:  # noqa: BLE001
                r["errors"].append({"stage": f"e2e/{name}", "trace": traceback.format_exc()[-3000:]})
            save()
    problems.clear()
    gc.collect()
    gpu.free()
    if "resources" in stages:
        r["resource_usage"] = profiling.resource_usage()
        save()
    if "ncu" in stages:
        r["ncu"] = ncu_stage(a.ncu_size)
        save()
    r["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    save()
    print("wrote", out, "errors:", len(r["errors"]), flush=True)


if __name__ == "__main__":
    main()
