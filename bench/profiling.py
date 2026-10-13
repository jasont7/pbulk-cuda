"""Device utilisation metrics that can be collected on Colab: nvidia-smi sampling, the register and
stack usage ptxas reports for each kernel, and Nsight Compute or nvprof when they are installed.
Each collector returns a dict with "available" and never raises, so a missing tool cannot stop a run.
"""
import csv
import datetime as dt
import glob
import io
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SMI_FIELDS = ("utilization.gpu", "utilization.memory", "memory.used", "power.draw")
RAW_LIMIT = 6000


def parse_smi(text, t0=None, t1=None):
    """Rows of `nvidia-smi --query-gpu=timestamp,... --format=csv,noheader,nounits`.

    Keeps the samples taken between t0 and t1 (datetime) when given. Returns one list per field.
    """
    cols = {f: [] for f in SMI_FIELDS}
    n_all = 0
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(SMI_FIELDS) + 1:
            continue
        try:
            stamp = dt.datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f")
            vals = [float(p) for p in parts[1:]]
        except ValueError:
            continue
        n_all += 1
        if (t0 is None or stamp >= t0) and (t1 is None or stamp <= t1):
            for f, v in zip(SMI_FIELDS, vals):
                cols[f].append(v)
    return cols, n_all


def smi_sample(fn, seconds=6.0, interval_ms=100):
    """Call fn() in a loop for `seconds` while nvidia-smi samples the device. fn must block until done."""
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "reason": "nvidia-smi not found"}
    try:
        proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp," + ",".join(SMI_FIELDS),
             "--format=csv,noheader,nounits", "-lms", str(interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        time.sleep(1.0)                      # let the sampler start
        t0 = dt.datetime.now()
        end = time.perf_counter() + seconds
        calls = 0
        while time.perf_counter() < end:
            fn()
            calls += 1
        t1 = dt.datetime.now()
        time.sleep(0.3)
        proc.terminate()
        text = proc.communicate(timeout=10)[0]
        # nvidia-smi averages over its own sample period, so drop the first half second of the window
        cols, n_all = parse_smi(text, t0 + dt.timedelta(seconds=0.5), t1)
        out = {"available": bool(cols[SMI_FIELDS[0]]), "seconds": seconds, "interval_ms": interval_ms,
               "calls": calls, "calls_per_s": calls / seconds, "samples_total": n_all,
               "samples": len(cols[SMI_FIELDS[0]])}
        for f, v in cols.items():
            if v:
                out[f] = {"median": float(np.median(v)), "mean": float(np.mean(v)),
                          "min": float(np.min(v)), "max": float(np.max(v))}
        if not out["available"]:
            out["reason"] = "no samples parsed"
            out["raw"] = text[:RAW_LIMIT]
        return out
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": repr(e)}


def parse_resource_usage(text):
    """Per-kernel numbers from `nvcc --resource-usage` (ptxas info lines)."""
    out = {}
    chunks = re.split(r"Compiling entry function '([^']+)'", text)
    for name, body in zip(chunks[1::2], chunks[2::2]):
        entry = {}
        for key, pat in (("registers", r"Used (\d+) registers"), ("stack_bytes", r"(\d+) bytes stack frame"),
                         ("spill_store_bytes", r"(\d+) bytes spill stores"),
                         ("spill_load_bytes", r"(\d+) bytes spill loads")):
            m = re.search(pat, body)
            if m:
                entry[key] = int(m.group(1))
        if entry:
            out[name] = entry
    return out


def resource_usage(arch="sm_75"):
    """Registers per thread and stack frame of each kernel, as compiled by nvcc for the T4."""
    nvcc = shutil.which("nvcc") or next(iter(sorted(glob.glob("/usr/local/cuda*/bin/nvcc"))), None)
    if nvcc is None:
        return {"available": False, "reason": "nvcc not found"}
    try:
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([nvcc, "--resource-usage", f"-arch={arch}", "-c", str(ROOT / "pbulk" / "kernels.cu"),
                                "-o", str(pathlib.Path(d) / "kernels.o")],
                               capture_output=True, text=True, timeout=300)
        text = r.stdout + r.stderr
        kernels = parse_resource_usage(text)
        return {"available": bool(kernels), "compiler": nvcc, "arch": arch, "kernels": kernels,
                "note": "compiled by nvcc. CuPy compiles the same source with NVRTC, so counts may differ slightly.",
                "raw": text[:RAW_LIMIT]}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": repr(e)}


def parse_ncu_csv(text):
    """Metrics per kernel from `ncu --csv` output. Returns {kernel: {metric: {"value", "unit"}}}."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if "Metric Name" in l and "Metric Value" in l), None)
    if start is None:
        return {}
    out = {}
    for row in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        name, metric = row.get("Kernel Name"), row.get("Metric Name")
        if not name or not metric:
            continue
        kernel = next((k for k in ("pb_scatter", "pb_gather") if k in name), name)
        # keep the first launch of each kernel
        out.setdefault(kernel, {}).setdefault(metric, {"value": row.get("Metric Value"),
                                                       "unit": row.get("Metric Unit")})
    return out


def _find(tool, patterns):
    return shutil.which(tool) or next((hit for p in patterns for hit in sorted(glob.glob(p))), None)


NCU_PATHS = ("/usr/local/cuda*/bin/ncu", "/opt/nvidia/nsight-compute/*/ncu",
             "/usr/local/cuda*/nsight-compute*/ncu")


def _try_install_ncu():
    """Best effort: Colab images ship the CUDA apt repository but not always Nsight Compute."""
    if shutil.which("apt-get") is None or shutil.which("nvcc") is None:
        return "apt-get or nvcc not found"
    try:
        v = re.search(r"release (\d+)\.(\d+)", subprocess.run(["nvcc", "--version"], capture_output=True,
                                                             text=True, timeout=60).stdout)
        if not v:
            return "could not read the CUDA version"
        pkg = f"cuda-nsight-compute-{v.group(1)}-{v.group(2)}"
        r = subprocess.run(["apt-get", "install", "-y", "-qq", pkg], capture_output=True, text=True, timeout=240)
        return f"apt-get install {pkg}: exit {r.returncode} {r.stderr[-300:]}"
    except Exception as e:  # noqa: BLE001
        return repr(e)


def ncu_profile(size="medium", install=True):
    """Nsight Compute on bench.profile_target. Colab may not allow the counters, so this can fail."""
    log = []
    ncu = _find("ncu", NCU_PATHS)
    if ncu is None and install:
        log.append(_try_install_ncu())
        ncu = _find("ncu", NCU_PATHS)
    if ncu is None:
        return {"available": False, "reason": "ncu not found", "log": log}
    target = [sys.executable, "-m", "bench.profile_target", "--size", size]
    for extra in (["--kernel-name", "regex:pb_", "--launch-count", "4"], ["--launch-count", "60"]):
        try:
            r = subprocess.run([ncu, "--csv", *extra, *target], capture_output=True, text=True,
                               timeout=420, cwd=str(ROOT))
            metrics = {k: v for k, v in parse_ncu_csv(r.stdout).items() if k in ("pb_scatter", "pb_gather")}
            log.append(f"ncu {' '.join(extra)}: exit {r.returncode}")
            if metrics:
                return {"available": True, "tool": ncu, "kernels": metrics, "log": log,
                        "raw": r.stdout[:RAW_LIMIT]}
            log.append((r.stdout + r.stderr)[-1500:])
        except Exception as e:  # noqa: BLE001
            log.append(repr(e))
    return {"available": False, "reason": "ncu ran but returned no kernel metrics", "log": log}


def nvprof_profile(size="medium"):
    """nvprof summary, if the toolkit still ships it. Stored as raw text only."""
    nvprof = _find("nvprof", ("/usr/local/cuda*/bin/nvprof",))
    if nvprof is None:
        return {"available": False, "reason": "nvprof not found"}
    try:
        r = subprocess.run([nvprof, sys.executable, "-m", "bench.profile_target", "--size", size],
                           capture_output=True, text=True, timeout=420, cwd=str(ROOT))
        text = r.stdout + r.stderr
        return {"available": r.returncode == 0 and "pb_gather" in text, "exit": r.returncode,
                "raw": text[-RAW_LIMIT:]}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": repr(e)}
