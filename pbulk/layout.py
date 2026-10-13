"""Host-side data layout for the donor-batched gather kernel (pb_gather in kernels.cu)."""
import numpy as np


def scatter_arrays(X, S):
    """The five arrays pb_scatter reads: S as CSR (pseudo-bulk -> cells), X as CSC (cell -> genes)."""
    X = X.tocsc()
    assert X.nnz < 2**31 - 1, "nnz must fit int32"
    return [np.ascontiguousarray(a, dtype=t) for a, t in (
        (S.indptr, np.int32), (S.indices, np.int32),
        (X.indptr, np.int32), (X.indices, np.int32), (X.data, np.float32))]


def gather_arrays(lay):
    """The four arrays pb_gather reads, from build_donor_layout."""
    return [lay[k] for k in ("seg_ptr", "col", "val", "mask")]


def build_donor_layout(X, S, subject, K, row_chunk=512):
    """Reorder cells by donor, store X as CSR, and index nonzeros by (gene, donor) segment.

    X: scipy sparse, genes x cells. S: CSR n_pb x cells from make_selection, rows ordered
    donor-major (donors sorted, K pseudo-bulks each). subject: donor label per cell.
    """
    n_genes, n_cells = X.shape
    n_pb = S.shape[0]
    assert K <= 32, "the per-cell mask is one uint32"
    assert n_pb % K == 0
    donors, codes = np.unique(np.asarray(subject), return_inverse=True)
    n_donors = donors.size
    assert n_pb == n_donors * K

    coo = S.tocoo()
    assert np.array_equal(codes[coo.col], coo.row // K), "pseudo-bulk rows must be donor-major"
    mask = np.zeros(n_cells, dtype=np.uint32)
    np.bitwise_or.at(mask, coo.col, (np.uint32(1) << (coo.row % K).astype(np.uint32)))

    perm = np.argsort(codes, kind="stable")
    Xr = X.tocsc()[:, perm].tocsr()
    Xr.sort_indices()
    codes_perm = codes[perm]

    counts = np.zeros(n_genes * n_donors, dtype=np.int64)
    for g0 in range(0, n_genes, row_chunk):
        g1 = min(g0 + row_chunk, n_genes)
        lo, hi = Xr.indptr[g0], Xr.indptr[g1]
        rows = np.repeat(np.arange(g1 - g0), np.diff(Xr.indptr[g0:g1 + 1]))
        key = rows * n_donors + codes_perm[Xr.indices[lo:hi]]
        counts[g0 * n_donors:g1 * n_donors] = np.bincount(key, minlength=(g1 - g0) * n_donors)
    seg_ptr = np.concatenate([[0], np.cumsum(counts)])
    assert seg_ptr[-1] == Xr.nnz < 2**31 - 1, "nnz must fit int32"

    return dict(seg_ptr=seg_ptr.astype(np.int32), col=Xr.indices.astype(np.int32),
                val=Xr.data.astype(np.float32), mask=mask[perm],
                n_genes=n_genes, n_donors=n_donors, K=K)


def gather_model(lay):
    """NumPy model of pb_gather, one (gene, donor) pair at a time. For tests on small data."""
    G, D, K = lay["n_genes"], lay["n_donors"], lay["K"]
    out = np.zeros((D * K, G), dtype=np.float64)
    bits = np.arange(K, dtype=np.uint32)
    for pair in range(G * D):
        g, d = divmod(pair, D)
        sl = slice(lay["seg_ptr"][pair], lay["seg_ptr"][pair + 1])
        v = lay["val"][sl].astype(np.float64)
        m = lay["mask"][lay["col"][sl]]
        out[d * K:(d + 1) * K, g] = v @ ((m[:, None] >> bits) & 1)
    return out
