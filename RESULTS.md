# Results (Oct 6, Colab Tesla T4, CuPy 14.0.1)

Synthetic data with the real data's structure (30 donors x K=30 = 900 pseudo-bulks, CPM columns).
Both CUDA kernels were checked against the CPU reference on the **same cell selection**.
GPU times are kernel + row normalisation, measured with CUDA events (median of 10), and exclude
the one-time upload and layout step.

| Problem | nnz | CPU loop | CPU batched (scipy) | CUDA scatter | CUDA gather | Max rel err (scatter / gather) |
|---|---|---|---|---|---|---|
| 4,000 genes x 50,000 cells | 16.0 M | 13,935 ms | 989 ms | 25.5 ms | **1.52 ms** | 4.9e-7 / 9.9e-7 |
| 8,000 genes x 120,000 cells | 76.8 M | not run | 4,389 ms | 246.5 ms | **4.90 ms** | 4.8e-7 / 1.2e-6 |

| Speedup | 16 M nnz | 76.8 M nnz |
|---|---|---|
| gather vs scatter | 16.8x | 50.3x |
| gather vs CPU batched | 652x | 896x |
| gather vs CPU loop (mirrors the R code) | 9,186x | |

Upload plus layout build (one-time, host side): 0.69 s and 3.2 s.

## Caveats

- Synthetic data, not the BAL dataset. Real nnz per cell and the cell-count spread per donor
  will differ. Re-run on the real export before quoting numbers for the real problem.
- The CPU baselines are scipy on Colab's CPU. The 17 min R figure was on Narval and is not
  measured here. The loop baseline mirrors the R code, not the R runtime.
- float32 on the GPU against a float64 CPU reference. Errors of 1e-6 relative are rounding.
  (Checked Oct 13: scipy keeps the float32 input type, so that reference also sums in float32 and only
  its normalisation is float64. The IR1 harness compares against a true float64 reference.)
- One run each. The GPU timings are medians over 10 repeats but the CPU ones are best of 3 (or 1).
- Timings exclude host-to-device transfer. The layout build (3.2 s at the larger size) costs
  more than the kernel. Worth reporting honestly in IR1.
