# pbulk-cuda

GPU generation of donor-stable pseudo-bulk RNA-seq profiles (CSC 586F project).

Each pseudo-bulk is the CPM sum of a random subset of one donor's cells, renormalised to 1e6.
That is `PB = S @ X^T` with `X` genes x cells (sparse) and `S` a 0/1 cell-selection matrix,
n_pb x cells. Real size: 14,378 genes x 241,924 cells, 30 donors x 30 = 900 pseudo-bulks.
The R original takes about 17 min and 48 GB on Narval.

```
pbulk/reference.py   CPU reference (loop = mirrors the R code, batched = one sparse product)
pbulk/seq_ref.cpp    sequential C++ reference, one thread (called through pbulk/cpu_cpp.py)
pbulk/kernels.cu     the two CUDA kernels, pb_scatter and pb_gather
pbulk/layout.py      host-side data layout for pb_gather
pbulk/gpu.py         CuPy host code (needs a GPU)
pbulk/synthetic.py   fake data with the real shape, so everything runs locally
tests/               correctness: loop = batched = dense, rows sum to 1e6, no cross-donor cells,
                     plus both kernels run on the CPU by an emulator (tests/emu)
bench/               timing harness. bench/ir1.py runs every IR1 experiment and writes one JSON
notebooks/           ir1_colab.ipynb, the Colab notebook that runs bench/ir1.py on a T4
report/              Interim Report 1 (LaTeX, figures, and the scripts that fill it from the JSON)
scripts/             one-time R export of the real data (run on Narval)
```

```
pip install -r requirements.txt
python -m pytest -q
python -m bench.run_reference --cells 50000
```

## Interim Report 1

1. Open `notebooks/ir1_colab.ipynb` in Colab, pick a T4 runtime, Run all. It downloads `ir1_results.json`.
2. Save that file as `report/data/ir1_results.json`.
3. `pip install -r requirements-dev.txt`, then `python report/build.py`. This fills the tables and the
   timing plot from the JSON and compiles `report/ir1.pdf`.

Without step 2 the report builds with the Oct 6 numbers from `RESULTS.md` and a red TBD for everything
not measured yet. `report/build.py` refuses a results file that did not come from a real GPU.

The PDF is compiled with [tectonic](https://tectonic-typesetting.github.io), kept out of git under `tools/`:

```
mkdir -p tools && curl -sL https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-aarch64-apple-darwin.tar.gz | tar xz -C tools
```

No tectonic: `python report/build.py` still writes `report/generated/results.tex` and the figures, and
the `report/` folder can be uploaded to Overleaf as it is.

After changing code under `pbulk/`, `bench/` or `tests/`, run `python notebooks/build_notebook.py`. The
notebook carries a snapshot of the code and a test fails when it is stale.

A local dry run of the harness with no GPU (kernels in the CPU emulator, timings meaningless):
`python -m bench.ir1 --backend emulator --tiny --out /tmp/dry.json`

Correctness rule for the CUDA version: same `S` in, same `PB` out (within float32 tolerance).

## Status

Both CUDA kernels (`pb_scatter`, `pb_gather` in `pbulk/kernels.cu`) match the CPU reference on a
Colab T4. `pb_gather` reads each nonzero once and feeds all K pseudo-bulks of a donor from
registers, which is 17x to 50x faster than the atomic scatter version. See `RESULTS.md`.
All of that is on synthetic data. Next: run the IR1 notebook, then profile with Nsight, try shared
memory and tensor cores, run on the real data.
