from bench.ir1 import CASES, SIZES
from pbulk.launch import gather_launch, regime, scatter_launch


def test_scatter_launch_is_one_block_per_pseudo_bulk():
    assert scatter_launch(900) == dict(grid=900, block=256, warps=7200, threads=230400)


def test_gather_launch_is_one_warp_per_gene_donor_pair():
    lay = gather_launch(4000, 30)
    assert lay == dict(grid=30000, block=128, warps=120000, threads=3840000, pairs=120000)
    assert gather_launch(7, 3)["grid"] == 6          # 21 pairs, 4 per block, last block partial


def test_regimes():
    assert regime(scatter_launch(1, 32)) == "one warp"
    assert regime(scatter_launch(1, 256)) == "one block, several warps"
    assert regime(scatter_launch(2, 32)) == "several blocks"
    assert regime(gather_launch(1, 1, 32)) == "one warp"
    assert regime(gather_launch(8, 4, 1024)) == "one block, several warps"
    assert regime(gather_launch(8, 4, 128)) == "several blocks"


def test_hand_picked_cases_cover_every_regime_for_both_kernels():
    want = {"one warp", "one block, several warps", "several blocks"}
    assert {regime(scatter_launch(c["donors"] * c["K"], c["sb"])) for c in CASES} == want
    assert {regime(gather_launch(c["genes"], c["donors"], c["gb"])) for c in CASES} == want


def test_benchmark_sizes_need_thousands_of_threads():
    for cfg in SIZES.values():
        assert gather_launch(cfg["genes"], cfg["donors"])["threads"] > 1_000_000
