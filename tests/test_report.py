"""The report builder: only real numbers get in, and both the before and after states render."""
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
pytestmark = pytest.mark.skipif(not (REPORT / "fill_results.py").exists(), reason="report/ not present")
pytest.importorskip("matplotlib")


def fill():
    import importlib.util
    import sys
    sys.path.insert(0, str(REPORT / "figures"))
    spec = importlib.util.spec_from_file_location("fill_results", REPORT / "fill_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_oct6_numbers_are_exactly_those_of_results_md():
    text = (ROOT / "RESULTS.md").read_text()
    o = json.loads((REPORT / "data" / "oct6_results.json").read_text())
    for size in o["sizes"].values():
        for key, value in size.items():
            if value is None:
                continue
            unit = {"gather_vs_scatter": "x", "gather_vs_batched": "x", "gather_vs_loop": "x",
                    "upload_layout_s": " s"}.get(key, "")
            assert re.search(r"(?<![\d.,])" + re.escape(value + unit) + r"(?![\d,])", text), (key, value)
    assert f"{o['date']}, Colab {o['gpu']}, CuPy {o['cupy']}" in text


def test_work_counts_file_matches_a_recount():
    from bench import ir1
    stored = json.loads((REPORT / "data" / "work_counts.json").read_text())
    X, S, subject, K = ir1.make_problem(ir1.SIZES["small"])
    w = ir1.work_counts(X, S, K, ir1.SIZES["small"]["donors"])
    assert {k: stored["small"][k] for k in w} == w
    assert set(stored) == set(ir1.SIZES)


def test_without_results_everything_unmeasured_is_tbd(tmp_path):
    m = fill().main(["--outdir", str(tmp_path)])
    tex = (tmp_path / "generated" / "results.tex").read_text()
    assert (tmp_path / "figures" / "fig_timing.pdf").stat().st_size > 1000
    for name in ("CorrectnessRows", "EndToEndRows", "ProfileRows", "EndToEndVerdict", "VsCppMedium"):
        assert r"\TBD" in m[name] and not re.search(r"\d{2,}|\d\.\d", m[name]), name      # no measured value
    assert m["AblationMedium"] == "16.8" and m["AblationLarge"] == "50.3" and m["GatherMsLarge"] == "4.90"
    assert "Oct 6" in m["ResSource"] and r"\newcommand{\KernelRows}" in tex
    assert "4.9e-7 and 4.8e-7 for scatter and 9.9e-7 and 1.2e-6 for gather" in m["FullSizeVerdict"]


def test_formatting_never_rounds_a_speedup_up():
    f = fill()
    assert f.fmt_x(9.99) == "9.9" and f.fmt_x(16.85) == "16.8" and f.fmt_x(896.9) == "896" and f.fmt_x(9186.7) == "9,186"
    assert f.fmt_ms(13935.2) == "13,935" and f.fmt_ms(246.54) == "246.5" and f.fmt_ms(4.9) == "4.90"


def test_refuses_results_that_did_not_come_from_a_gpu(tmp_path):
    from bench import ir1
    from tests import emu
    if not emu.available():
        pytest.skip("no C++ compiler")
    path = tmp_path / "r.json"
    ir1.main(["--backend", "emulator", "--tiny", "--profile-seconds", "0.2", "--out", str(path)])
    with pytest.raises(SystemExit, match="not a GPU"):
        fill().main([str(path), "--outdir", str(tmp_path / "out")])
    assert not (tmp_path / "out" / "generated" / "results.tex").exists()

    # The rendering code for real results, exercised on a relabelled copy. Output goes to a temp dir only.
    r = json.loads(path.read_text())
    r["meta"]["backend"] = "cuda"
    r["profile"]["smi"] = {k: {"available": True, "samples": 40, "utilization.gpu": {"median": 99.0},
                               "utilization.memory": {"median": 50.0}} for k in ("scatter", "gather")}
    r["profile"]["resource_usage"] = {"kernels": {"pb_scatter": {"registers": 14, "stack_bytes": 0},
                                                  "pb_gather": {"registers": 50, "stack_bytes": 0}}}
    fake = tmp_path / "relabelled.json"
    fake.write_text(json.dumps(r))
    m = fill().main([str(fake), "--outdir", str(tmp_path / "out")])
    assert r"\TBD" not in m["KernelRows"] + m["CorrectnessRows"] + m["EndToEndRows"] + m["EndToEndVerdict"]
    assert "99\\%" in m["ProfileRows"] and "50 \\\\" in m["ProfileRows"]
    # the emulated "gather" is slower than "scatter", so the guard on the report's prose must fire
    assert "Rewrite that paragraph" in m["NotebookStatus"] and "matched the CPU reference" in m["FullSizeVerdict"]
    for line in m["CorrectnessRows"].splitlines()[:3]:
        got, total = re.findall(r"(\d+) of (\d+)", line)[0]
        assert got == total and int(total) > 0
