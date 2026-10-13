"""Where the kernels run. `cuda` is the real device (pbulk.gpu). `emulator` runs the same kernel source
on the CPU (tests/emu), so the harness can be tested without a GPU. Emulator timings mean nothing and
the results file says which backend produced it.
"""
import time

import numpy as np


class CudaBackend:
    name = "cuda"

    def __init__(self):
        from pbulk import gpu
        self.gpu = gpu
        if getattr(gpu.cp, "IS_FAKE", False):     # the test stand-in in tests/fake_cupy
            self.name = "fake-cupy"
        for f in ("to_device", "to_host", "sync", "free", "scatter_kernel", "gather_kernel",
                  "time_samples", "device_info"):
            setattr(self, f, getattr(gpu, f))


class EmulatorBackend:
    name = "emulator"

    def __init__(self):
        from tests import emu
        self.emu = emu

    def to_device(self, arrays):
        return list(arrays)

    def to_host(self, a):
        return np.asarray(a)

    def sync(self):
        pass

    def free(self):
        pass

    @staticmethod
    def _normalise(PB, normalise):
        return PB * (np.float32(1e6) / PB.sum(axis=1, keepdims=True)) if normalise else PB

    def scatter_kernel(self, arrays, n_pb, n_genes, block=256, normalise=True):
        return self._normalise(self.emu.scatter(arrays, n_pb, n_genes, block), normalise)

    def gather_kernel(self, arrays, n_genes, n_donors, K, block=128, normalise=True):
        return self._normalise(self.emu.gather(arrays, n_genes, n_donors, K, block), normalise)

    def time_samples(self, fn, repeats=10, warmup=2):
        times = []
        for _ in range(repeats):
            t = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t) * 1e3)
        return times

    def device_info(self):
        return {"name": "CPU emulator (not a GPU)"}


def get(name):
    return {"cuda": CudaBackend, "emulator": EmulatorBackend}[name]()
