"""A stand-in for CuPy so the tests can import and run pbulk/gpu.py on a machine with no GPU.

Arrays are NumPy arrays and the two kernels run in the CPU emulator (tests/emu). Only what
pbulk/gpu.py uses is here. IS_FAKE makes the harness label its output "fake-cupy", and the
report builder refuses anything that is not labelled "cuda".
"""
import time
import types

import numpy as np

IS_FAKE = True
__version__ = "fake"
float32 = np.float32
asarray = np.asarray
asnumpy = np.asarray
zeros = np.zeros
empty = np.empty


class _Kernel:
    def __init__(self, name):
        self.name = name

    def __call__(self, grid, block, args, shared_mem=0):
        from pbulk.launch import gather_launch, scatter_launch
        from tests import emu
        if self.name == "pb_scatter":
            *arrays, PB, n_genes = args
            assert len(arrays) == 5 and grid == (scatter_launch(PB.shape[0], block[0])["grid"],)
            PB += emu.scatter(arrays, PB.shape[0], int(n_genes), block[0])
        elif self.name != "pb_gather":
            *arrays, PB, n_genes, n_donors, K = args
            PB[...] = emu.gather_variant(self.name[3:], arrays, int(n_genes), int(n_donors), int(K), block[0])
        else:
            *arrays, PB, n_genes, n_donors, K = args
            assert len(arrays) == 4 and PB.shape == (n_donors * K, n_genes)
            assert grid == (gather_launch(int(n_genes), int(n_donors), block[0])["grid"],)
            PB[...] = emu.gather(arrays, int(n_genes), int(n_donors), int(K), block[0])


class RawModule:
    def __init__(self, code):
        assert "pb_scatter" in code and "pb_gather" in code

    def get_function(self, name):
        assert name in ("pb_scatter", "pb_gather", "pb_gather_xr", "pb_gather_fm", "pb_gather_cs")
        return _Kernel(name)


class _Event:
    def record(self):
        self.t = time.perf_counter()

    def synchronize(self):
        pass


class _Device:
    def synchronize(self):
        pass


class _Pool:
    def free_all_blocks(self):
        pass


def get_default_memory_pool():
    return _Pool()


cuda = types.SimpleNamespace(
    Device=_Device, Event=_Event,
    get_elapsed_time=lambda a, b: (b.t - a.t) * 1e3,
    runtime=types.SimpleNamespace(getDeviceProperties=lambda i: {
        "name": b"FAKE cupy (CPU emulator, not a GPU)", "multiProcessorCount": 0, "maxThreadsDim": (0, 0, 0)}))


def __getattr__(name):          # anything else (searchsorted, bincount, ...) is NumPy's
    return getattr(np, name)
