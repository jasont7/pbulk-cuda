"""IR2: the GPU layout (run here with NumPy/SciPy) matches the host layout, and the three new kernels
(run on the CPU emulator) match the float64 reference."""
import numpy as np
import pytest
import scipy.sparse as sp

from bench.ir1 import make_case, reference64
from pbulk.layout import (build_donor_layout, build_donor_layout_xp, complement_arrays, complement_work,
                          donor_codes, fused_arrays, gather_arrays)
from tests import emu

CASES = [dict(genes=7, cells=300, donors=3, K=30, density=0.3, seed=1),
         dict(genes=40, cells=900, donors=5, K=32, density=0.1, seed=2),
         dict(genes=13, cells=200, donors=1, K=5, density=0.5, seed=3, ratio=(0.05, 0.95)),
         dict(genes=9, cells=150, donors=4, K=1, density=0.4, seed=4),
         dict(genes=25, cells=500, donors=6, K=17, density=0.2, seed=5, lone_donor=True)]


def layouts(case):
    X, S, subject = make_case(**case)
    K = case["K"]
    host = build_donor_layout(X, S, subject, K)
    codes, D = donor_codes(subject)
    Xc = X.tocsc()
    Xc.indices = Xc.indices.astype(np.int32)
    xp_lay = build_donor_layout_xp(np, sp, Xc, S.indptr, S.indices, codes, D, K)
    return X, S, host, xp_lay


@pytest.mark.parametrize("case", CASES)
def test_xp_layout_matches_host(case):
    _, _, host, lay = layouts(case)
    for k in ("seg_ptr", "col", "val", "mask", "donor_perm", "donor_cells", "pb_size"):
        assert np.array_equal(np.asarray(host[k]), np.asarray(lay[k])), k


def test_complement_work_counts():
    _, _, host, _ = layouts(CASES[0])
    w = complement_work(host)
    assert w["complement_adds"] <= w["masked_adds"]
    assert w["lane_steps_cs"] <= w["lane_steps_baseline"] + w["warp_iterations"]


pytestmark_emu = pytest.mark.skipif(not emu.available(), reason="no C++ compiler")


@pytestmark_emu
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("block", [32, 128])
@pytest.mark.parametrize("which", ["gather_xr", "gather_fm", "gather_cs"])
def test_variant_kernels_emulated(case, block, which):
    X, S, host, _ = layouts(case)
    ref = reference64(X, S)
    D, K, G = host["n_donors"], host["K"], X.shape[0]
    arrays = {"gather_xr": gather_arrays(host), "gather_fm": fused_arrays(host),
              "gather_cs": complement_arrays(host)}[which]
    out = emu.gather_variant(which, arrays, G, D, K, block).astype(np.float64)
    # the complement subtracts two sums, so allow absolute error relative to the segment totals
    tol = np.abs(ref).max() * 1e-5
    assert np.allclose(out, ref, rtol=1e-4, atol=tol), np.abs(out - ref).max()
