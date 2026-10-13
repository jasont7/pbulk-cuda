"""Builds and calls the sequential C++ reference in seq_ref.cpp (one thread, -O3)."""
import ctypes
import pathlib
import shutil
import subprocess
import tempfile

import numpy as np

from .layout import scatter_arrays

SRC = pathlib.Path(__file__).parent / "seq_ref.cpp"
FLAGS = ["-O3", "-std=c++17", "-shared", "-fPIC"]
_lib = None


def available():
    return shutil.which("c++") is not None


def _load():
    global _lib
    if _lib is None:
        out = pathlib.Path(tempfile.mkdtemp(prefix="pbulk_")) / "libseq_ref.so"
        subprocess.run(["c++", *FLAGS, "-o", str(out), str(SRC)], check=True)
        _lib = ctypes.CDLL(str(out))
        ptr = np.ctypeslib.ndpointer
        _lib.pb_sequential.argtypes = [ptr(np.int32), ptr(np.int32), ptr(np.int32), ptr(np.int32),
                                       ptr(np.float32), ptr(np.float32), ctypes.c_int, ctypes.c_int]
        _lib.pb_sequential.restype = None
    return _lib


def prepare(X, S):
    """Convert once, outside any timed region. Returns the arguments of compute()."""
    _load()
    return scatter_arrays(X, S), S.shape[0], X.shape[0]


def compute(args):
    """Run the sequential reference. Returns the normalised float32 PB (n_pb x n_genes)."""
    arrays, n_pb, n_genes = args
    PB = np.empty((n_pb, n_genes), dtype=np.float32)
    _load().pb_sequential(*arrays, PB, n_pb, n_genes)
    return PB
