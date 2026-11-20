"""Writes notebooks/ir2_colab.ipynb: the same code snapshot as the IR1 notebook, running bench/ir2.py.

    python notebooks/build_ir2_notebook.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_notebook as nb  # noqa: E402

OUT = nb.ROOT / "notebooks" / "ir2_colab.ipynb"

INTRO = """# pbulk-cuda: Interim Report 2 experiments

**How to run:** Runtime > Change runtime type > **T4 GPU**, then Runtime > **Run all**. About fifteen minutes.

Runs the unit tests, then `bench/ir2.py`: the GPU layout build against the host one (idea 1), the five
kernels and cuSPARSE on three synthetic sizes, a block-size sweep, end to end with transfers, and Nsight
Compute on the medium size. Writes `ir2_results.json` and offers it as a download.
"""

HARNESS = """harness_code, _ = run([sys.executable, "-m", "bench.ir2", "--out", RESULTS, *EXTRA_ARGS])
print("harness exit code", harness_code)
"""

SUMMARY = """import json
results = json.loads(RESULTS.read_text())
results["tests"] = {"pytest_exit": test_code, "summary": test_summary}
RESULTS.write_text(json.dumps(results, indent=1))
print("errors:", len(results["errors"]))
for e in results["errors"]:
    print(e["stage"]); print(e["trace"][-800:])
print("ncu available:", results.get("ncu", {}).get("available"))
"""


PRINT = """# The results file as gzip + base64 between markers, for copying out of the page if the download is blocked.
import base64, gzip
print("BEGIN_IR2_RESULTS")
print(base64.b64encode(gzip.compress(RESULTS.read_bytes())).decode())
print("END_IR2_RESULTS")
"""


def build():
    snap = "# The code of the repository, packed by notebooks/build_notebook.py. Generated, do not edit.\n" \
           f'SNAPSHOT = "{nb.snapshot()}"\n'
    unpack = nb.UNPACK.replace("pbulk-cuda-ir1", "pbulk-cuda-ir2").replace("ir1_results.json", "ir2_results.json")
    cells = [nb.cell("markdown", INTRO), nb.cell("code", nb.CHECK_GPU), nb.cell("code", snap), nb.cell("code", unpack),
             nb.cell("code", nb.DEPS), nb.cell("code", nb.TESTS), nb.cell("code", HARNESS), nb.cell("code", SUMMARY),
             nb.cell("code", PRINT), nb.cell("code", nb.DOWNLOAD)]
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i}"
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}}}


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
