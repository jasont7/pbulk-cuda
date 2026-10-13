"""The IR1 harness, end to end, without a GPU. Timings from these runs are never used."""
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from bench import backends, ir1, profiling
from tests import emu

ROOT = pathlib.Path(__file__).resolve().parent.parent
needs_cxx = pytest.mark.skipif(not emu.available(), reason="no C++ compiler")


@needs_cxx
def test_correctness_sweep_passes_on_the_emulated_kernels():
    res = ir1.correctness(backends.get("emulator"), n_random=4, big=False)
    bad = [(c["name"], c["integer"], k) for c in res["cases"] for k in ("scatter", "gather") if not c[k]["ok"]]
    assert not bad and res["all_ok"]
    assert res["n_cases"] == 2 * (len(ir1.CASES) + 4)
    for k in ("scatter", "gather"):
        assert res["regimes_passed"][k] == ["one block, several warps", "one warp", "several blocks"]
        assert all(c[k]["exact"] for c in res["cases"] if c["integer"])     # integer sums are exact


def test_compare_rejects_wrong_output():
    ref = np.array([[2.0, 0.0], [3.0, 5.0]])
    assert ir1.compare(ref.astype(np.float32), ref, integer=True)["ok"]
    assert not ir1.compare(ref + [[0, 0], [0, 1]], ref, integer=True)["ok"]
    assert not ir1.compare(ref * 1.001, ref, integer=False)["ok"]
    assert not ir1.compare(ref + [[0, 1e-9], [0, 0]], ref, integer=False)["ok"]    # a zero must stay zero


def test_work_counts():
    X, S, subject, K = ir1.make_problem(dict(genes=50, cells=600, donors=3, K=4))
    w = ir1.work_counts(X, S, K, 3)
    dense = (X.toarray() != 0).astype(int)
    assert w["terms"] == int((S.toarray().astype(int) @ dense.T).sum())      # one addition per term of S X^T
    assert w["gather_masked_adds"] == 32 * X.nnz and w["segments"] == 150


@needs_cxx
def test_full_run_on_the_emulator(tmp_path):
    out = tmp_path / "r.json"
    ir1.main(["--backend", "emulator", "--tiny", "--profile-seconds", "0.2", "--out", str(out)])
    r = json.loads(out.read_text())
    assert r["meta"]["backend"] == "emulator" and not r["errors"]
    assert set(r["kernel"]) == {"small", "medium", "large"}
    for k in r["kernel"].values():
        assert k["scatter_ok"] and k["gather_ok"] and len(k["gather_ms"]) == k["trials"]
    e = r["end_to_end"]
    assert e["size"] == "medium" and len(e["gather"]["total_ms"]) == e["repeats"]
    assert r["profile"]["size"] == "large" and set(r["profile"]["smi"]) == {"scatter", "gather"}
    assert "NOT A GPU RUN" in ir1.summary(r)


@needs_cxx
@pytest.mark.parametrize("module,args", [
    ("bench.ir1", ["--tiny", "--stages", "correctness,kernel"]),
    ("bench.run_gpu", ["--genes", "8", "--cells", "300", "--donors", "2", "--K", "3"]),
    ("bench.profile_target", ["--tiny"]),
])
def test_gpu_host_code_runs_against_fake_cupy(tmp_path, module, args):
    """pbulk/gpu.py cannot be imported without CuPy. tests/fake_cupy stands in for it."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "tests" / "fake_cupy"), str(ROOT)]))
    if module != "bench.profile_target":
        args = args + ["--out", str(tmp_path / "r.json")]
    p = subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "FAIL" not in p.stdout
    if module == "bench.ir1":
        r = json.loads((tmp_path / "r.json").read_text())
        assert r["meta"]["backend"] == "fake-cupy" and not r["errors"] and r["correctness"]["all_ok"]


def test_parse_smi_keeps_the_window():
    text = ("2026/10/13 10:00:00.100, 0, 0, 300, 26.10\n"
            "2026/10/13 10:00:01.100, 98, 61, 1900, 69.52\n"
            "2026/10/13 10:00:02.100, 100, 63, 1900, 70.01\n"
            "garbage line\n"
            "2026/10/13 10:00:09.100, 3, 0, 1900, 30.00\n")
    cols, n = profiling.parse_smi(text, dt.datetime(2026, 10, 13, 10, 0, 1), dt.datetime(2026, 10, 13, 10, 0, 3))
    assert n == 4 and cols["utilization.gpu"] == [98.0, 100.0] and cols["utilization.memory"] == [61.0, 63.0]


def test_parse_resource_usage():
    text = """ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'pb_scatter' for 'sm_75'
ptxas info    : Function properties for pb_scatter
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, 400 bytes cmem[0]
ptxas info    : Compiling entry function 'pb_gather' for 'sm_75'
ptxas info    : Function properties for pb_gather
    8 bytes stack frame, 4 bytes spill stores, 4 bytes spill loads
ptxas info    : Used 50 registers, 392 bytes cmem[0]
"""
    got = profiling.parse_resource_usage(text)
    assert got["pb_scatter"] == dict(registers=14, stack_bytes=0, spill_store_bytes=0, spill_load_bytes=0)
    assert got["pb_gather"]["registers"] == 50 and got["pb_gather"]["stack_bytes"] == 8


def test_parse_ncu_csv():
    text = ('some banner\n"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream",'
            '"Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"\n'
            '"0","1","python","h","pb_gather","1","7","(128, 1, 1)","(30000, 1, 1)","0","7.5",'
            '"Occupancy","Achieved Occupancy","","71.2"\n'
            '"1","1","python","h","pb_gather","1","7","(128, 1, 1)","(30000, 1, 1)","0","7.5",'
            '"Occupancy","Achieved Occupancy","","70.0"\n')
    got = profiling.parse_ncu_csv(text)
    assert got == {"pb_gather": {"Achieved Occupancy": {"value": "71.2", "unit": ""}}}
    assert profiling.parse_ncu_csv("no table here") == {}


def test_missing_tools_do_not_raise():
    if profiling.shutil.which("nvidia-smi") is None:
        assert profiling.smi_sample(lambda: None, seconds=0.1)["available"] is False
    assert isinstance(profiling.resource_usage()["available"], bool)
