"""CPU reference for the donor-stable pseudo-bulk generator.

Python port of BioResearch/pipeline/scripts/generate_pseudo_bulks.R.

For each donor d, K pseudo-bulks are made by sampling a random subset of d's cells
(ratio ~ U[ratio_min, ratio_max]), summing their CPM columns, and renormalising to 1e6.
Cells are never pooled across donors.

Sampling is separate from computation. `make_selection` draws which cells go in which
pseudo-bulk and stores it as a CSR 0/1 matrix S (n_pb x n_cells). Every compute path takes
the same S, so the CPU loop, the batched CPU version and the CUDA kernel can be compared
on identical inputs. (R's RNG can't be reproduced in numpy, so the R CSV can only be
compared statistically, never cell for cell.)
"""
import numpy as np
import scipy.sparse as sp

CT_LEVELS = ("cell_type1", "cell_type2", "cell_type3")


def make_selection(subject, K=30, ratio_min=0.3, ratio_max=1.0, seed=43):
    """Return (S, pb_donor, ratio). S is CSR, n_pb x n_cells, entries 0/1."""
    rng = np.random.default_rng(seed)
    subject = np.asarray(subject)
    donors = np.unique(subject)
    cells_of = {d: np.flatnonzero(subject == d) for d in donors}
    indptr, indices, pb_donor, ratios = [0], [], [], []
    for d in donors:
        cells = cells_of[d]
        for _ in range(K):
            ratio = rng.uniform(ratio_min, ratio_max)
            n_sub = max(1, int(ratio * cells.size))
            sel = np.sort(rng.choice(cells, size=n_sub, replace=False))
            indices.append(sel)
            indptr.append(indptr[-1] + n_sub)
            pb_donor.append(d)
            ratios.append(ratio)
    indices = np.concatenate(indices).astype(np.int32)
    data = np.ones(indices.size, dtype=np.float32)
    S = sp.csr_matrix((data, indices, np.array(indptr, dtype=np.int64)),
                      shape=(len(pb_donor), subject.size))
    return S, np.array(pb_donor), np.array(ratios)


def _renorm(PB):
    return PB / PB.sum(axis=1, keepdims=True) * 1e6


def compute_loop(X, S):
    """Mirrors the R code: one indicator vector and one sparse mat-vec per pseudo-bulk."""
    n_pb, n_cells = S.shape
    PB = np.zeros((n_pb, X.shape[0]), dtype=np.float64)
    for i in range(n_pb):
        indicator = np.zeros(n_cells, dtype=X.dtype)
        indicator[S.indices[S.indptr[i]:S.indptr[i + 1]]] = 1
        PB[i] = X @ indicator
    return _renorm(PB)


def compute_batched(X, S):
    """All pseudo-bulks at once: PB = S @ X^T, a sparse x sparse product."""
    Xt = X.T.tocsr()
    PB = (S @ Xt).toarray().astype(np.float64)
    return _renorm(PB)


def fractions(S, labels):
    """Cell-type fractions of each pseudo-bulk for one label vector."""
    cats, codes = np.unique(np.asarray(labels), return_inverse=True)
    onehot = sp.csr_matrix((np.ones(codes.size, dtype=np.float32),
                            (np.arange(codes.size), codes)), shape=(codes.size, cats.size))
    counts = (S @ onehot).toarray()
    return counts / counts.sum(axis=1, keepdims=True), cats
