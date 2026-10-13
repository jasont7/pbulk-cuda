"""Writes notebooks/ir1_colab.ipynb. The notebook carries a snapshot of the code, so it runs on Colab
with Run all and no repository URL. Rerun this after changing anything under pbulk/, bench/ or tests/
(tests/test_notebook.py fails when the snapshot is stale).

    python notebooks/build_notebook.py
"""
import base64
import io
import json
import pathlib
import tarfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "ir1_colab.ipynb"
PATTERNS = ["requirements.txt", "pbulk/*.py", "pbulk/*.cu", "pbulk/*.cpp", "bench/*.py",
            "tests/*.py", "tests/emu/*", "tests/fake_cupy/cupy/*.py"]


def snapshot_files():
    files = sorted({p for pat in PATTERNS for p in ROOT.glob(pat) if p.is_file() and "__pycache__" not in p.parts})
    return [p.relative_to(ROOT).as_posix() for p in files]


def snapshot():
    """The code as a base64 tar.gz. Fixed timestamps, so the same sources give the same notebook."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9, format=tarfile.PAX_FORMAT) as tar:
        for name in snapshot_files():
            data = (ROOT / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    raw = buf.getvalue()
    raw = raw[:4] + b"\0\0\0\0" + raw[8:]          # zero the gzip header timestamp
    return base64.b64encode(raw).decode()


INTRO = """# pbulk-cuda: Interim Report 1 experiments

**How to run:** Runtime > Change runtime type > **T4 GPU**, then Runtime > **Run all**. No edits needed. It takes roughly fifteen minutes.

What it does, in order:

1. checks that a GPU is attached and unpacks the code (a snapshot stored in this notebook)
2. runs the unit tests (CPU references, kernel emulator)
3. **correctness on the GPU**: both kernels against a float64 reference on instances that need one warp, one block of several warps and several blocks, plus edge cases and a random sweep
4. **kernel time without transfers** on three sizes, against the single-threaded CPU baselines. Scatter against gather is the layout ablation (3 trials of 10 runs each, medians)
5. **end-to-end time** on the medium size, including host preparation, host-to-device copies, kernel launch and the copy back
6. **device utilisation**: nvidia-smi sampling, register and stack usage from ptxas, and Nsight Compute or nvprof if this runtime has them
7. writes everything to `ir1_results.json`, prints a summary table and offers the file as a download

Afterwards: save the file as `report/data/ir1_results.json` in the repository and run `python report/build.py`. That fills the tables and plots of the report.

All data is **synthetic** (`pbulk/synthetic.py`). The real dataset is not used here.
"""

CHECK_GPU = """import os, pathlib, shutil, subprocess, sys

# Leave empty to use the code stored in this notebook. Set to a git URL to clone the repository instead.
REPO_URL = ""

# Only for testing this notebook on a machine without a GPU. Leave unset on Colab.
EXTRA_ARGS = os.environ.get("PBULK_IR1_ARGS", "").split()
NO_GPU_OK = bool(EXTRA_ARGS)

if shutil.which("nvidia-smi") is None and not NO_GPU_OK:
    raise SystemExit("No GPU attached. Runtime > Change runtime type > T4 GPU, then Run all.")
if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                         capture_output=True, text=True).stdout)


def run(cmd, **kw):
    \"\"\"Run a command and show its output as it comes.\"\"\"
    print("$", " ".join(map(str, cmd)), flush=True)
    p = subprocess.Popen(list(map(str, cmd)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)
    lines = []
    for line in p.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    return p.wait(), "".join(lines)
"""

UNPACK = """import base64, io, tarfile

WORK = pathlib.Path(os.environ.get("PBULK_IR1_WORKDIR", "/content/pbulk-cuda-ir1"))
if REPO_URL:
    if not (WORK / ".git").exists():
        code, _ = run(["git", "clone", "--depth", "1", REPO_URL, WORK])
        assert code == 0, "git clone failed"
    print("using a clone of", REPO_URL)
else:
    WORK.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(SNAPSHOT)), mode="r:gz") as tar:
        tar.extractall(WORK)
    print("unpacked the code snapshot stored in this notebook into", WORK)
os.chdir(WORK)
RESULTS = WORK / "results" / "ir1_results.json"
print(sorted(p.name for p in WORK.iterdir()))
"""

DEPS = """# Colab ships CuPy, NumPy, SciPy and pandas. Install only what is missing.
import importlib.util

need = [m for m in ("numpy", "scipy", "pandas", "pytest") if importlib.util.find_spec(m) is None]
if importlib.util.find_spec("cupy") is None and not NO_GPU_OK:
    need.append("cupy-cuda12x")
if need:
    run([sys.executable, "-m", "pip", "install", "-q", *need])
if not NO_GPU_OK:
    import cupy
    print("CuPy", cupy.__version__, "on", cupy.cuda.runtime.getDeviceProperties(0)["name"].decode())
"""

TESTS = """# Unit tests: CPU references, the C++ baseline, and both kernels in the CPU emulator.
test_code, test_out = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"])
test_summary = test_out.strip().splitlines()[-1] if test_out.strip() else ""
print("pytest exit code", test_code)
"""

HARNESS = """# The experiments. Progress is printed as each stage finishes, and the JSON is saved after every stage.
harness_code, _ = run([sys.executable, "-m", "bench.ir1", "--out", RESULTS, *EXTRA_ARGS])
print("harness exit code", harness_code)
"""

SUMMARY = """import json

sys.path.insert(0, str(WORK))
from bench.ir1 import summary

results = json.loads(RESULTS.read_text())
results["tests"] = {"pytest_exit": test_code, "summary": test_summary}
RESULTS.write_text(json.dumps(results, indent=1))
print(summary(results))
print("\\nunit tests:", test_summary)
if results["errors"]:
    print("\\nSome stages failed. The rest of the file is still valid. Rerun, or report what is above.")
"""

DOWNLOAD = """print("results file:", RESULTS)
try:
    from google.colab import files
    files.download(str(RESULTS))
except Exception as e:      # not on Colab
    print("no browser download here:", type(e).__name__)
"""

OUTRO = """## Next

Save the downloaded file as `report/data/ir1_results.json` in the repository, then run

```
python report/build.py
```

It refuses a results file that was not produced on a real GPU.
"""


def cell(kind, source):
    c = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def build():
    snap = "# The code of the repository, packed by notebooks/build_notebook.py. Generated, do not edit.\n" \
           f'SNAPSHOT = "{snapshot()}"\n'
    cells = [cell("markdown", INTRO), cell("code", CHECK_GPU), cell("code", snap), cell("code", UNPACK),
             cell("code", DEPS), cell("code", TESTS), cell("code", HARNESS), cell("code", SUMMARY),
             cell("code", DOWNLOAD), cell("markdown", OUTRO)]
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i}"
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}}}


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {len(snapshot_files())} files in the snapshot)")
