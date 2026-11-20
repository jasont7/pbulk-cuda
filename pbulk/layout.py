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
                donor_perm=codes_perm.astype(np.int32), donor_cells=np.bincount(codes, minlength=n_donors),
                pb_size=np.diff(S.indptr).astype(np.int64),
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


# ---------------------------------------------------------------------------------------------------------
# IR2
# ---------------------------------------------------------------------------------------------------------

def donor_codes(subject):
    """Donor code 0..D-1 per cell (donors in sorted order, as build_donor_layout numbers them)."""
    donors, codes = np.unique(np.asarray(subject), return_inverse=True)
    return codes.astype(np.int32), donors.size


def flip_bits(lay, xp=np):
    """Idea 2. Bit k of flip[d] is set when pseudo-bulk k of donor d takes more than half the donor's cells.
    For those the kernel adds the unselected cells and subtracts from the segment total."""
    D, K = lay["n_donors"], lay["K"]
    big = (2 * lay["pb_size"].reshape(D, K) > lay["donor_cells"].reshape(D, 1)).astype(xp.int64)
    return (big << xp.arange(K, dtype=xp.int64)).sum(axis=1).astype(xp.uint32)


def fused_arrays(lay):
    """Idea 3, pb_gather_fm: seg_ptr, the mask of each nonzero's cell (mask[col]) and val."""
    return [lay["seg_ptr"], lay["mask"][lay["col"]], lay["val"]]


def complement_arrays(lay, xp=np):
    """Idea 2, pb_gather_cs: seg_ptr, mask ^ flip of each nonzero's cell, val, flip per donor."""
    flip = flip_bits(lay, xp)
    cell_bits = lay["mask"] ^ flip[lay["donor_perm"]]
    return [lay["seg_ptr"], cell_bits[lay["col"]], lay["val"], flip]


def complement_work(lay):
    """Adds the complement kernel does: per nonzero 1 + popcount(mask ^ flip), and per warp iteration the
    MAX over its 32 lanes (lanes wait for the slowest). Host layout (NumPy). Counted, not measured."""
    _, bits, _, flip = complement_arrays(lay)
    pc = np.zeros(bits.size, dtype=np.int64)
    for k in range(32):
        pc += (bits >> k) & 1
    seg = lay["seg_ptr"].astype(np.int64)
    n = seg[1:] - seg[:-1]
    pair = np.repeat(np.arange(n.size), n)
    pos = np.arange(bits.size) - seg[:-1][pair]
    iters = (n + 31) // 32
    it_base = np.concatenate([[0], np.cumsum(iters)[:-1]])
    it_id = it_base[pair] + pos // 32
    n_it = int(iters.sum())
    wmax = np.zeros(n_it, dtype=np.int64)
    np.maximum.at(wmax, it_id, pc)
    return dict(nnz=int(bits.size), mean_set_bits=float(pc.mean()),
                flipped_pseudo_bulks=int(sum(bin(int(f)).count("1") for f in flip)),
                complement_adds=int(pc.sum() + bits.size), masked_adds=int(32 * bits.size),
                warp_iterations=n_it, lane_steps_baseline=int(32 * n_it),
                lane_steps_cs=int(wmax.sum() + n_it), mean_warp_max_set_bits=float(wmax.mean()) if n_it else 0.0,
                mean_unflipped_set_bits=float(np.mean([bin(int(m)).count("1") for m in lay["mask"][:2000]])))


def build_donor_layout_xp(xp, sps, X_csc, s_indptr, s_indices, codes, n_donors, K):
    """Idea 1. The same arrays as build_donor_layout, built with array module xp (numpy, or cupy on the GPU)
    and sparse module sps (scipy.sparse or cupyx.scipy.sparse). No Python loop over genes or donors.

    X_csc: sps.csc_matrix genes x cells, float32, int32 indices. s_indptr / s_indices: the CSR arrays of S
    (rows donor-major). codes: donor code per cell, int32.
    """
    G, C = X_csc.shape
    nsel = int(s_indices.size)
    # mask: bit (row % K) of each selected cell. Bits are distinct, so their sum is their OR, exact in float64.
    rows = xp.searchsorted(s_indptr, xp.arange(nsel), side="right") - 1
    w = xp.left_shift(xp.ones(nsel, dtype=xp.int64), (rows % K).astype(xp.int64)).astype(xp.float64)
    mask = xp.bincount(s_indices, weights=w, minlength=C).astype(xp.int64).astype(xp.uint32)
    # cells sorted by donor, ties by cell id (unique keys, so any sort is stable)
    perm = xp.argsort(codes.astype(xp.int64) * C + xp.arange(C, dtype=xp.int64))
    # permute the columns of X: a segmented gather
    indptr = X_csc.indptr.astype(xp.int64)
    lens = (indptr[1:] - indptr[:-1])[perm]
    nptr = xp.concatenate([xp.zeros(1, xp.int64), xp.cumsum(lens)])
    nnz = int(X_csc.nnz)
    j = xp.arange(nnz, dtype=xp.int64)
    newcol = xp.searchsorted(nptr, j, side="right") - 1
    src = indptr[:-1][perm][newcol] + (j - nptr[newcol])
    Xp = sps.csc_matrix((X_csc.data[src], X_csc.indices[src], nptr.astype(xp.int32)), shape=(G, C))
    del newcol, src
    Xr = Xp.tocsr()                       # the transpose: cuSPARSE on the GPU
    Xr.sort_indices()
    codes_perm = codes[perm]
    rp = Xr.indptr.astype(xp.int64)
    row = xp.searchsorted(rp, j, side="right") - 1
    key = row * n_donors + codes_perm[Xr.indices].astype(xp.int64)   # nondecreasing: rows sorted, cells donor-sorted
    seg_ptr = xp.searchsorted(key, xp.arange(G * n_donors + 1, dtype=xp.int64), side="left")
    pb_size = (s_indptr[1:] - s_indptr[:-1]).astype(xp.int64)
    return dict(seg_ptr=seg_ptr.astype(xp.int32), col=Xr.indices.astype(xp.int32), val=Xr.data.astype(xp.float32),
                mask=mask[perm], donor_perm=codes_perm.astype(xp.int32),
                donor_cells=xp.bincount(codes, minlength=n_donors), pb_size=pb_size,
                n_genes=G, n_donors=n_donors, K=K)
