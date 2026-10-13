"""Launch geometry of the two kernels. Plain Python, so it can be checked without a GPU."""
WARP = 32
# A block may use at most 65,536 registers. pb_gather needs 65 per thread (ptxas, Colab T4), so a
# 1024-thread block asks for 66,560 and the launch fails with CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES.
GATHER_MAX_BLOCK = 512


def scatter_launch(n_pb, block=256):
    """pb_scatter: one block per pseudo-bulk, block // 32 warps each."""
    assert block % WARP == 0 and 0 < block <= 1024
    return dict(grid=n_pb, block=block, warps=n_pb * (block // WARP), threads=n_pb * block)


def gather_launch(n_genes, n_donors, block=128):
    """pb_gather: one warp per (gene, donor) pair, block // 32 pairs per block."""
    assert block % WARP == 0 and 0 < block <= GATHER_MAX_BLOCK
    per_block = block // WARP
    n_pairs = n_genes * n_donors
    grid = (n_pairs + per_block - 1) // per_block
    return dict(grid=grid, block=block, warps=grid * per_block, threads=grid * block, pairs=n_pairs)


def regime(launch):
    """Which rung of the thread hierarchy a launch needs: one warp, one block, or several blocks."""
    if launch["grid"] > 1:
        return "several blocks"
    return "one warp" if launch["block"] == WARP else "one block, several warps"
