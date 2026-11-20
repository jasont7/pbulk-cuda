"""For Nsight Compute: each IR2 kernel launched once on one input (python -m bench.ir2_profile_target)."""
import argparse

from bench.ir1 import SIZES, make_problem

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="medium", choices=sorted(SIZES))
    a = ap.parse_args()

    import cupy as cp
    from pbulk import gpu
    from pbulk.layout import build_donor_layout, complement_arrays, fused_arrays, gather_arrays
    X, S, subject, K = make_problem(SIZES[a.size])
    G, D = X.shape[0], S.shape[0] // K
    lay = build_donor_layout(X, S, subject, K)
    base = gpu.to_device(gather_arrays(lay))
    lay_d = {k: (cp.asarray(v) if hasattr(v, "dtype") else v) for k, v in lay.items()}
    fm, cs = fused_arrays(lay_d), complement_arrays(lay_d, cp)
    gpu.gather_kernel(base, G, D, K)
    gpu.variant_kernel("gather_xr", base, G, D, K)
    gpu.variant_kernel("gather_fm", fm, G, D, K)
    gpu.variant_kernel("gather_cs", cs, G, D, K)
    gpu.sync()
    print("profile target done")
