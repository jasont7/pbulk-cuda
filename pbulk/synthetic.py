"""Synthetic single-cell data with the same shape and sparsity as the BAL dataset.

Real data: 14378 genes x 241924 cells, 30 donors, 3 cell-type levels, CPM values.
This lets the reference and the CUDA kernel run anywhere, without the 2.6 GB RDS.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp


def make_synthetic(n_genes=2000, n_cells=20000, n_donors=30, density=0.08,
                   n_types=(6, 12, 17), seed=0):
    rng = np.random.default_rng(seed)
    nnz_per_cell = max(1, int(density * n_genes))
    rows = np.concatenate([rng.choice(n_genes, nnz_per_cell, replace=False)
                           for _ in range(n_cells)])
    cols = np.repeat(np.arange(n_cells), nnz_per_cell)
    vals = rng.lognormal(mean=0.0, sigma=1.0, size=rows.size)
    X = sp.csc_matrix((vals, (rows, cols)), shape=(n_genes, n_cells))
    X = X @ sp.diags(1e6 / np.asarray(X.sum(axis=0)).ravel())  # CPM: each cell sums to 1e6
    X = X.tocsc().astype(np.float32)

    donor = rng.integers(0, n_donors, n_cells)
    meta = pd.DataFrame({"SubjectName": [f"donor{d:02d}" for d in donor]})
    for level, n in enumerate(n_types, start=1):
        meta[f"cell_type{level}"] = [f"ct{level}_{t}" for t in rng.integers(0, n, n_cells)]
    return X, meta
