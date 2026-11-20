"""CuPy host code for the CUDA kernels in kernels.cu. Needs a GPU, so it is not imported by the tests."""
import pathlib

import cupy as cp
import numpy as np

from .launch import gather_launch, scatter_launch
from .layout import build_donor_layout, gather_arrays, scatter_arrays

_mod = cp.RawModule(code=(pathlib.Path(__file__).parent / "kernels.cu").read_text())
_scatter = _mod.get_function("pb_scatter")
_gather = _mod.get_function("pb_gather")


def sync():
    cp.cuda.Device().synchronize()


def to_device(arrays):
    """Host-to-device copy of a list of NumPy arrays. Returns once the copies are done."""
    out = [cp.asarray(a) for a in arrays]
    sync()
    return out


def to_host(a):
    return cp.asnumpy(a)


def free():
    cp.get_default_memory_pool().free_all_blocks()


class GpuData:
    """Everything the kernels read, copied to the device once (transfers are timed separately)."""

    def __init__(self, X, S, subject, K, layout=None):
        self.n_genes, self.n_cells = X.shape
        self.n_pb, self.K = S.shape[0], K
        self.scatter = to_device(scatter_arrays(X, S))
        lay = layout if layout is not None else build_donor_layout(X, S, subject, K)
        self.n_donors = lay["n_donors"]
        self.gather = to_device(gather_arrays(lay))


def _normalise(PB):
    PB *= (1e6 / PB.sum(axis=1, keepdims=True))
    return PB


def scatter_kernel(arrays, n_pb, n_genes, block=256, normalise=True):
    launch = scatter_launch(n_pb, block)
    PB = cp.zeros((n_pb, n_genes), dtype=cp.float32)
    _scatter((launch["grid"],), (block,), (*arrays, PB, np.int32(n_genes)))
    return _normalise(PB) if normalise else PB


def gather_kernel(arrays, n_genes, n_donors, K, block=128, normalise=True):
    launch = gather_launch(n_genes, n_donors, block)
    PB = cp.empty((n_donors * K, n_genes), dtype=cp.float32)
    _gather((launch["grid"],), (block,), (*arrays, PB, np.int32(n_genes), np.int32(n_donors), np.int32(K)))
    return _normalise(PB) if normalise else PB


def run_scatter(d, block=256, normalise=True):
    return scatter_kernel(d.scatter, d.n_pb, d.n_genes, block, normalise)


def run_gather(d, block=128, normalise=True):
    return gather_kernel(d.gather, d.n_genes, d.n_donors, d.K, block, normalise)


def time_samples(fn, repeats=10, warmup=2):
    """Milliseconds of each of `repeats` calls of fn(), measured with CUDA events."""
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(repeats):
        a, b = cp.cuda.Event(), cp.cuda.Event()
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(float(cp.cuda.get_elapsed_time(a, b)))
    return times


def time_gpu(fn, repeats=10, warmup=2):
    """Median milliseconds of fn() measured with CUDA events."""
    return float(np.median(time_samples(fn, repeats, warmup)))


def device_info():
    """Name and limits of device 0 as plain Python values, for the results file."""
    props = cp.cuda.runtime.getDeviceProperties(0)
    info = {}
    for k, v in props.items():
        if isinstance(v, bytes):
            info[k] = v.decode(errors="replace")
        elif isinstance(v, (int, float, str)):
            info[k] = v
        elif isinstance(v, (tuple, list)):
            info[k] = [int(x) for x in v]
    info["cupy"] = cp.__version__
    return info


# ---------------------------------------------------------------------------------------------------------
# IR2
# ---------------------------------------------------------------------------------------------------------
from .layout import build_donor_layout_xp  # noqa: E402

_variants = {k: _mod.get_function("pb_" + k) for k in ("gather_xr", "gather_fm", "gather_cs")}
CS_STRIDE = 33


def variant_kernel(which, arrays, n_genes, n_donors, K, block=128, normalise=True):
    """pb_gather_xr (arrays from gather_arrays), pb_gather_fm (fused_arrays), pb_gather_cs (complement_arrays)."""
    launch = gather_launch(n_genes, n_donors, block)
    PB = cp.empty((n_donors * K, n_genes), dtype=cp.float32)
    shared = (block // 32) * 32 * CS_STRIDE * 4 if which == "gather_cs" else 0
    _variants[which]((launch["grid"],), (block,), (*arrays, PB, np.int32(n_genes), np.int32(n_donors), np.int32(K)),
                     shared_mem=shared)
    return _normalise(PB) if normalise else PB


class GpuRaw:
    """The inputs of the GPU layout build: X as CSC, S's CSR arrays, donor code per cell, on the device."""

    def __init__(self, X, S, codes):
        import cupyx.scipy.sparse as csp
        Xc = X.tocsc()
        self.X = csp.csc_matrix((cp.asarray(Xc.data.astype(np.float32)), cp.asarray(Xc.indices.astype(np.int32)),
                                 cp.asarray(Xc.indptr.astype(np.int32))), shape=Xc.shape)
        self.s_indptr = cp.asarray(S.indptr.astype(np.int64))
        self.s_indices = cp.asarray(S.indices.astype(np.int64))
        self.codes = cp.asarray(codes.astype(np.int32))
        sync()

    def nbytes(self):
        return int(self.X.data.nbytes + self.X.indices.nbytes + self.X.indptr.nbytes + self.s_indptr.nbytes
                   + self.s_indices.nbytes + self.codes.nbytes)


def layout_gpu(raw, n_donors, K):
    """Idea 1: build_donor_layout on the device (cupy + cuSPARSE)."""
    import cupyx.scipy.sparse as csp
    return build_donor_layout_xp(cp, csp, raw.X, raw.s_indptr, raw.s_indices, raw.codes, n_donors, K)


def cusparse_operands(X, S):
    """Library baseline: S (CSR, float32) and X^T (CSR) on the device for cuSPARSE SpGEMM."""
    import cupyx.scipy.sparse as csp
    S32 = csp.csr_matrix(S.astype(np.float32))
    XT = csp.csr_matrix(X.T.tocsr().astype(np.float32))
    return S32, XT


def cusparse_pb(S32, XT, normalise=True):
    PB = (S32 @ XT).toarray()
    return _normalise(PB) if normalise else PB
