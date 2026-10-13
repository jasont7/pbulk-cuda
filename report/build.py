"""One command to rebuild the report: fill the tables and plots, redraw the figures, compile the PDF.

    python report/build.py                                    # uses report/data/ir1_results.json if it exists
    python report/build.py --results path/to/ir1_results.json

Needs a LaTeX engine. This looks for tools/tectonic (see README) and then for tectonic on the PATH.
Without one it still writes generated/results.tex and the figures, so report/ can be uploaded to Overleaf.
"""
import argparse
import os
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "figures"))


def page_count(pdf):
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(pdf)).pages)
    except Exception:  # noqa: BLE001
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=None)
    a = ap.parse_args(argv)

    default = HERE / "data" / "ir1_results.json"
    results = a.results or (str(default) if default.exists() else None)
    import fill_results
    import fig_parallelism
    fill_results.main([results] if results else [])
    fig_parallelism.main()

    local = ROOT / "tools" / "tectonic"
    tectonic = str(local) if local.exists() else shutil.which("tectonic")
    if tectonic is None:
        print("No LaTeX engine found. generated/results.tex and the figures are up to date.\n"
              "Either put the tectonic binary at tools/tectonic (README), or upload report/ to Overleaf.")
        return 1
    env = dict(os.environ, XDG_CACHE_HOME=str(ROOT / "tools" / "cache"))
    r = subprocess.run([tectonic, "ir1.tex"], cwd=HERE, env=env, capture_output=True, text=True)
    notes = [l for l in (r.stdout + r.stderr).splitlines() if "warning" in l.lower() or "error" in l.lower()]
    print("\n".join(notes[-25:]))
    if r.returncode != 0:
        print(r.stderr[-3000:])
        return r.returncode
    n = page_count(HERE / "ir1.pdf")
    print(f"built {HERE / 'ir1.pdf'}" + (f", {n} pages" if n else ""))
    if n and n > 6:
        print("WARNING: over the six page limit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
