"""Run the CUDA kernels from pbulk/kernels.cu on the CPU. See shim.h for what this does and does not model."""
import pathlib
import shutil
import subprocess
import tempfile

import numpy as np

from pbulk.launch import gather_launch, scatter_launch

HERE = pathlib.Path(__file__).parent
_exe = None


def available():
    return shutil.which("c++") is not None


def _build():
    global _exe
    if _exe is None:
        out = pathlib.Path(tempfile.mkdtemp(prefix="pbulk_emu_")) / "run"
        subprocess.run(["c++", "-O1", "-std=c++17", "-Wno-unknown-pragmas", "-Wno-deprecated-declarations",
                        "-o", str(out), str(HERE / "run_kernel.cpp")], check=True)
        _exe = out
    return _exe


def _run(which, arrays, n_pb, n_genes, block, extra=()):
    exe = _build()
    with tempfile.TemporaryDirectory(prefix="pbulk_emu_") as d:
        for i, a in enumerate(arrays):
            np.ascontiguousarray(a).tofile(pathlib.Path(d) / f"a{i}.bin")
        subprocess.run([str(exe), which, d, str(block), str(n_pb), str(n_genes), *map(str, extra)], check=True)
        return np.fromfile(pathlib.Path(d) / "PB.f32", dtype=np.float32).reshape(n_pb, n_genes)


def scatter(arrays, n_pb, n_genes, block=256):
    """pb_scatter on the CPU. arrays from pbulk.layout.scatter_arrays. Returns unnormalised float32 PB."""
    scatter_launch(n_pb, block)
    return _run("scatter", arrays, n_pb, n_genes, block)


def gather(arrays, n_genes, n_donors, K, block=128):
    """pb_gather on the CPU. arrays from pbulk.layout.gather_arrays. Returns unnormalised float32 PB."""
    gather_launch(n_genes, n_donors, block)
    return _run("gather", arrays, n_donors * K, n_genes, block, (n_donors, K))
