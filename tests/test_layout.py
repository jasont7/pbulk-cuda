import numpy as np
from pbulk.layout import build_donor_layout, gather_model
from pbulk.reference import make_selection, compute_batched
from pbulk.synthetic import make_synthetic


def test_gather_model_matches_reference():
    X, meta = make_synthetic(n_genes=120, n_cells=1500, n_donors=4, density=0.15, seed=3)
    subject = meta["SubjectName"].values
    S, _, _ = make_selection(subject, K=6, seed=5)
    lay = build_donor_layout(X, S, subject, K=6)
    raw = gather_model(lay)
    raw = raw / raw.sum(axis=1, keepdims=True) * 1e6
    assert np.allclose(raw, compute_batched(X, S), rtol=1e-4)


def test_segments_cover_all_nonzeros_in_order():
    X, meta = make_synthetic(n_genes=60, n_cells=800, n_donors=3, density=0.2, seed=4)
    subject = meta["SubjectName"].values
    S, _, _ = make_selection(subject, K=5, seed=6)
    lay = build_donor_layout(X, S, subject, K=5)
    assert lay["seg_ptr"][0] == 0 and lay["seg_ptr"][-1] == X.nnz
    assert (np.diff(lay["seg_ptr"]) >= 0).all()
