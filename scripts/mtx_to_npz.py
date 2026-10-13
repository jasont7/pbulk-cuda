"""cpm.mtx (text, slow to read) -> cpm.npz (fast). Usage: python scripts/mtx_to_npz.py data/"""
import sys, os
import scipy.io, scipy.sparse as sp

d = sys.argv[1]
X = scipy.io.mmread(os.path.join(d, "cpm.mtx")).tocsc().astype("float32")
sp.save_npz(os.path.join(d, "cpm.npz"), X)
print("saved", X.shape, "nnz", X.nnz)
