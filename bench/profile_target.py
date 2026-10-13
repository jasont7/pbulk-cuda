"""A short program for command-line profilers (ncu, nvprof): each kernel launched twice on one input.

    ncu --csv --kernel-name regex:pb_ python -m bench.profile_target --size medium
"""
import argparse

from bench.ir1 import SIZES, TINY_SIZES, make_problem

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="medium", choices=sorted(SIZES))
    ap.add_argument("--tiny", action="store_true", help="tiny sizes, for the tests")
    a = ap.parse_args()

    from pbulk import gpu
    X, S, subject, K = make_problem((TINY_SIZES if a.tiny else SIZES)[a.size])
    d = gpu.GpuData(X, S, subject, K)
    for _ in range(2):
        gpu.run_scatter(d)
        gpu.run_gather(d)
    gpu.sync()
    print("profile target done")
