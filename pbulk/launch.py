"""Launch geometry of the two kernels. Plain Python, so it can be checked without a GPU."""
WARP = 32


def scatter_launch(n_pb, block=256):
    """pb_scatter: one block per pseudo-bulk, block // 32 warps each."""
    assert block % WARP == 0 and 0 < block <= 1024
    return dict(grid=n_pb, block=block, warps=n_pb * (block // WARP), threads=n_pb * block)


def gather_launch(n_genes, n_donors, block=128):
    """pb_gather: one warp per (gene, donor) pair, block // 32 pairs per block."""
    assert block % WARP == 0 and 0 < block <= 1024
    per_block = block // WARP
    n_pairs = n_genes * n_donors
    grid = (n_pairs + per_block - 1) // per_block
    return dict(grid=grid, block=block, warps=grid * per_block, threads=grid * block, pairs=n_pairs)


def regime(launch):
    """Which rung of the thread hierarchy a launch needs: one warp, one block, or several blocks."""
    if launch["grid"] > 1:
        return "several blocks"
    return "one warp" if launch["block"] == WARP else "one block, several warps"
