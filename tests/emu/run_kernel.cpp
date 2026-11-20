// Runs pb_scatter or pb_gather from kernels.cu on the CPU (see shim.h) for the tests.
//   run scatter <dir> <block> <n_pb> <n_genes>
//   run gather  <dir> <block> <n_pb> <n_genes> <n_donors> <K>
// Reads the kernel's input arrays from <dir>/a0.bin, a1.bin, ... and writes <dir>/PB.f32 (unnormalised).
#include "shim.h"   // first, see the note on _XOPEN_SOURCE
#include <cstring>
#include <functional>
#include <string>
#include <vector>
extern "C" { float smem[1 << 16]; }   // the dynamic shared memory of pb_gather_cs. Warps run one at a time
#include "../../pbulk/kernels.cu"

static std::vector<char> load(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> v(n);
    if (n && fread(v.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "bad %s\n", p.c_str()); exit(1); }
    fclose(f); return v;
}

static std::function<void()> kernel;
static void lane_main() { kernel(); emu::finished[emu::cur] = true; }

// One warp: resume every lane in turn until all have returned. Lanes that stop at a shuffle must all stop there.
static void run_warp(unsigned block, unsigned warp, std::vector<std::vector<char>>& stacks) {
    using namespace emu;
    for (int l = 0; l < 32; ++l) {
        finished[l] = waiting[l] = false;
        getcontext(&lane_ctx[l]);
        lane_ctx[l].uc_stack.ss_sp = stacks[l].data();
        lane_ctx[l].uc_stack.ss_size = stacks[l].size();
        lane_ctx[l].uc_link = &main_ctx;
        makecontext(&lane_ctx[l], lane_main, 0);
    }
    for (;;) {
        int n_done = 0, n_wait = 0;
        for (int l = 0; l < 32; ++l) {
            if (!finished[l]) {
                cur = l; waiting[l] = false;
                blockIdx.x = block; threadIdx.x = warp * 32 + l;
                swapcontext(&main_ctx, &lane_ctx[l]);
            }
            n_done += finished[l]; n_wait += waiting[l];
        }
        if (n_done == 32) return;
        if (n_wait != 32) { fprintf(stderr, "warp diverged at a shuffle\n"); exit(2); }
        for (int l = 0; l < 32; ++l) {
            if (offset[l] != offset[0] || xor_mode[l] != xor_mode[0]) { fprintf(stderr, "shuffles differ\n"); exit(2); }
            if (xor_mode[0]) res[l] = slot[l ^ offset[0]];
            else res[l] = (l + offset[0] < 32) ? slot[l + offset[0]] : slot[l];
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 6) { fprintf(stderr, "usage: see top of file\n"); return 1; }
    const std::string which = argv[1], dir = argv[2];
    const int block = atoi(argv[3]), n_pb = atoi(argv[4]), n_genes = atoi(argv[5]);
    std::vector<std::vector<char>> a;
    const int n_in = which == "scatter" ? 5 : which == "gather_fm" ? 3 : 4;
    for (int i = 0; i < n_in; ++i) a.push_back(load(dir + "/a" + std::to_string(i) + ".bin"));
    std::vector<float> PB((size_t)n_pb * n_genes, which == "scatter" ? 0.f : -1.f);  // gather must overwrite all of it

    blockDim.x = block;
    if (which == "scatter") {
        gridDim.x = n_pb;
        kernel = [&] { pb_scatter((const int*)a[0].data(), (const int*)a[1].data(), (const int*)a[2].data(),
                                  (const int*)a[3].data(), (const float*)a[4].data(), PB.data(), n_genes); };
    } else if (which != "gather") {
        const int n_donors = atoi(argv[6]), K = atoi(argv[7]), per_block = block / 32;
        gridDim.x = (n_genes * n_donors + per_block - 1) / per_block;
        if (which == "gather_xr")
            kernel = [&, n_donors, K] { pb_gather_xr((const int*)a[0].data(), (const int*)a[1].data(), (const float*)a[2].data(),
                                                     (const unsigned*)a[3].data(), PB.data(), n_genes, n_donors, K); };
        else if (which == "gather_fm")
            kernel = [&, n_donors, K] { pb_gather_fm((const int*)a[0].data(), (const unsigned*)a[1].data(),
                                                     (const float*)a[2].data(), PB.data(), n_genes, n_donors, K); };
        else if (which == "gather_cs")
            kernel = [&, n_donors, K] { pb_gather_cs((const int*)a[0].data(), (const unsigned*)a[1].data(),
                                                     (const float*)a[2].data(), (const unsigned*)a[3].data(),
                                                     PB.data(), n_genes, n_donors, K); };
        else { fprintf(stderr, "unknown kernel %s\n", which.c_str()); return 1; }
    } else {
        const int n_donors = atoi(argv[6]), K = atoi(argv[7]), per_block = block / 32;
        gridDim.x = (n_genes * n_donors + per_block - 1) / per_block;
        kernel = [&, n_donors, K] { pb_gather((const int*)a[0].data(), (const int*)a[1].data(), (const float*)a[2].data(),
                                              (const unsigned*)a[3].data(), PB.data(), n_genes, n_donors, K); };
    }
    std::vector<std::vector<char>> stacks(32, std::vector<char>(1 << 18));
    for (unsigned b = 0; b < gridDim.x; ++b)
        for (unsigned w = 0; w < blockDim.x / 32; ++w) run_warp(b, w, stacks);

    FILE* f = fopen((dir + "/PB.f32").c_str(), "wb");
    fwrite(PB.data(), sizeof(float), PB.size(), f); fclose(f);
    return 0;
}
