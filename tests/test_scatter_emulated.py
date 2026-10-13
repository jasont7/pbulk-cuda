"""Runs the kernels from kernels.cu as plain C++ (tests/emu) and checks them against the reference."""
import numpy as np
import pytest

from pbulk.layout import build_donor_layout, gather_arrays, scatter_arrays
from pbulk.reference import make_selection, compute_batched
from pbulk.synthetic import make_synthetic
from tests import emu

pytestmark = pytest.mark.skipif(not emu.available(), reason="no C++ compiler")


def problem():
    X, meta = make_synthetic(n_genes=150, n_cells=1200, n_donors=4, density=0.12, seed=8)
    subject = meta["SubjectName"].values
    S, _, _ = make_selection(subject, K=5, seed=9)
    return X, S, subject


def normalised(PB):
    PB = PB.astype(np.float64)
    return PB / PB.sum(axis=1, keepdims=True) * 1e6


def test_scatter_kernel_emulated():
    X, S, _ = problem()
    PB = emu.scatter(scatter_arrays(X, S), S.shape[0], X.shape[0])
    assert np.allclose(normalised(PB), compute_batched(X, S), rtol=1e-4)


def test_gather_kernel_emulated():
    X, S, subject = problem()
    lay = build_donor_layout(X, S, subject, K=5)
    PB = emu.gather(gather_arrays(lay), X.shape[0], lay["n_donors"], 5)
    assert np.allclose(normalised(PB), compute_batched(X, S), rtol=1e-4)


def test_scatter_and_gather_agree_for_any_block_size():
    X, S, subject = problem()
    X = X[:20]
    lay = build_donor_layout(X, S, subject, K=5)
    ref = emu.gather(gather_arrays(lay), 20, lay["n_donors"], 5, block=32)
    # gather stops at 512: a 1024-thread block exceeds the register limit on the device (pbulk/launch.py)
    for sblock, gblock in ((64, 64), (256, 256), (1024, 512)):
        assert np.array_equal(emu.gather(gather_arrays(lay), 20, lay["n_donors"], 5, block=gblock), ref)
        assert np.allclose(emu.scatter(scatter_arrays(X, S), S.shape[0], 20, block=sblock), ref, rtol=1e-5)
