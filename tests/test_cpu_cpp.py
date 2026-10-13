import numpy as np
import pytest

from pbulk import cpu_cpp
from pbulk.reference import make_selection, compute_batched
from pbulk.synthetic import make_synthetic


@pytest.mark.skipif(not cpu_cpp.available(), reason="no C++ compiler")
def test_sequential_cpp_matches_reference():
    X, meta = make_synthetic(n_genes=300, n_cells=2000, n_donors=5, density=0.1, seed=1)
    S, _, _ = make_selection(meta["SubjectName"].values, K=4, seed=7)
    PB = cpu_cpp.compute(cpu_cpp.prepare(X, S))
    assert PB.dtype == np.float32 and PB.shape == (20, 300)
    assert np.allclose(PB, compute_batched(X, S), rtol=1e-4)
    assert np.allclose(PB.sum(axis=1), 1e6, rtol=1e-5)
