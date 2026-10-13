"""The Colab notebook: valid, in sync with the sources, and runnable end to end (here on the CPU emulator)."""
import base64
import importlib.util
import io
import json
import pathlib
import tarfile

import pytest

from tests import emu

ROOT = pathlib.Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "ir1_colab.ipynb"
pytestmark = pytest.mark.skipif(not NB.exists(), reason="notebook not present (running from its own snapshot)")


def builder():
    spec = importlib.util.spec_from_file_location("build_notebook", ROOT / "notebooks" / "build_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def code_cells():
    return ["".join(c["source"]) for c in json.loads(NB.read_text())["cells"] if c["cell_type"] == "code"]


def test_notebook_is_valid_and_has_no_outputs():
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(str(NB), as_version=4)
    nbformat.validate(nb)
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")
    assert nb.metadata["accelerator"] == "GPU"


def test_notebook_is_what_the_builder_makes():
    assert json.loads(NB.read_text()) == builder().build(), "run: python notebooks/build_notebook.py"


def test_snapshot_matches_the_working_tree():
    ns = {}
    exec(next(c for c in code_cells() if c.lstrip().startswith("# The code of the repository")), ns)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(ns["SNAPSHOT"])), mode="r:gz") as tar:
        packed = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    assert sorted(packed) == builder().snapshot_files()
    for name, data in packed.items():
        assert data == (ROOT / name).read_bytes(), f"{name} is stale, run: python notebooks/build_notebook.py"
    for needed in ("pbulk/kernels.cu", "pbulk/gpu.py", "bench/ir1.py", "tests/emu/shim.h", "requirements.txt"):
        assert needed in packed


@pytest.mark.skipif(not emu.available(), reason="no C++ compiler")
def test_notebook_cells_run_top_to_bottom(tmp_path, monkeypatch):
    """Run all code cells in order, with the emulator standing in for the GPU. The unit-test cell is
    replaced, because running the test suite from inside itself would never finish."""
    monkeypatch.setenv("PBULK_IR1_ARGS", "--backend emulator --tiny --stages correctness,kernel")
    monkeypatch.setenv("PBULK_IR1_WORKDIR", str(tmp_path / "work"))
    monkeypatch.chdir(tmp_path)
    ns = {"__name__": "__main__"}
    for src in code_cells():
        if "pytest" in src and "test_code" in src and "bench.ir1" not in src:
            src = "test_code, test_summary = 0, 'skipped inside the test suite'"
        exec(compile(src, "<notebook cell>", "exec"), ns)
    r = json.loads((tmp_path / "work" / "results" / "ir1_results.json").read_text())
    assert r["meta"]["backend"] == "emulator" and not r["errors"]
    assert r["correctness"]["all_ok"] and r["tests"]["pytest_exit"] == 0
    assert (tmp_path / "work" / "pbulk" / "kernels.cu").exists()
