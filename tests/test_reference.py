import numpy as np
from pbulk.reference import make_selection, compute_loop, compute_batched, fractions
from pbulk.synthetic import make_synthetic


def small():
    X, meta = make_synthetic(n_genes=300, n_cells=2000, n_donors=5, density=0.1, seed=1)
    S, pb_donor, ratio = make_selection(meta["SubjectName"].values, K=4, seed=7)
    return X, meta, S, pb_donor, ratio


def test_loop_matches_batched():
    X, _, S, _, _ = small()
    assert np.allclose(compute_loop(X, S), compute_batched(X, S), rtol=1e-5)


def test_matches_dense_naive():
    X, _, S, _, _ = small()
    dense = X.toarray().astype(np.float64)
    ref = np.stack([dense[:, S.indices[S.indptr[i]:S.indptr[i + 1]]].sum(axis=1)
                    for i in range(S.shape[0])])
    ref = ref / ref.sum(axis=1, keepdims=True) * 1e6
    assert np.allclose(compute_batched(X, S), ref, rtol=1e-5)


def test_rows_sum_to_one_million():
    X, _, S, _, _ = small()
    assert np.allclose(compute_batched(X, S).sum(axis=1), 1e6)


def test_cells_never_cross_donors():
    _, meta, S, pb_donor, _ = small()
    subject = meta["SubjectName"].values
    for i in range(S.shape[0]):
        cells = S.indices[S.indptr[i]:S.indptr[i + 1]]
        assert set(subject[cells]) == {pb_donor[i]}


def test_subset_size_and_ratio_range():
    _, meta, S, pb_donor, ratio = small()
    assert ((ratio >= 0.3) & (ratio <= 1.0)).all()
    subject = meta["SubjectName"].values
    for i in range(S.shape[0]):
        n_d = (subject == pb_donor[i]).sum()
        assert S.indptr[i + 1] - S.indptr[i] == max(1, int(ratio[i] * n_d))


def test_fractions_sum_to_one():
    _, meta, S, _, _ = small()
    for level in ("cell_type1", "cell_type2", "cell_type3"):
        fr, _ = fractions(S, meta[level].values)
        assert np.allclose(fr.sum(axis=1), 1.0)
