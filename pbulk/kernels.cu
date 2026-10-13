// CUDA kernels for pseudo-bulk generation. PB[p, g] = sum of X[g, c] over the cells c selected for pseudo-bulk p.
// Row normalisation to 1e6 is done on the host side with two cheap elementwise ops (see gpu.py).
// Ptr arrays are int32, so nnz must stay below 2^31 (the host asserts this).

extern "C" {

// ---------------------------------------------------------------------------
// v1: scatter. One block per pseudo-bulk, one warp per selected cell.
// X is CSC (genes x cells): cell c's nonzeros are col_ptr[c]..col_ptr[c+1], rows in row_idx.
// Lanes stride over a cell's nonzeros and atomicAdd into PB. Every pseudo-bulk re-reads
// each of its cells from global memory, and atomics contend on common genes.
// ---------------------------------------------------------------------------
__global__ void pb_scatter(const int* __restrict__ sel_ptr, const int* __restrict__ sel_idx,
                           const int* __restrict__ col_ptr, const int* __restrict__ row_idx,
                           const float* __restrict__ val, float* PB, int n_genes)
{
    const int p = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n_warps = blockDim.x >> 5;
    float* out = PB + (size_t)p * n_genes;
    for (int i = sel_ptr[p] + warp; i < sel_ptr[p + 1]; i += n_warps) {
        const int c = sel_idx[i];
        for (int j = col_ptr[c] + lane; j < col_ptr[c + 1]; j += 32)
            atomicAdd(&out[row_idx[j]], val[j]);
    }
}

// ---------------------------------------------------------------------------
// v2: donor-batched gather. A cell belongs to one donor, so it can only appear in that donor's
// K pseudo-bulks. Cells are reordered by donor and X is stored CSR (genes x cells), so each
// (gene, donor) pair owns a contiguous segment of nonzeros: seg_ptr[g * n_donors + d].
// mask[c] has bit k set when cell c is in pseudo-bulk k of its donor (so K <= 32).
// One warp per (gene, donor): each nonzero is read ONCE and added to all K accumulators,
// instead of once per pseudo-bulk. No atomics. Lanes then combine their accumulators with shuffles.
// Output is pseudo-bulk-major: PB[(d * K + k) * n_genes + g].
// ---------------------------------------------------------------------------
__global__ void pb_gather(const int* __restrict__ seg_ptr, const int* __restrict__ col,
                          const float* __restrict__ val, const unsigned* __restrict__ mask,
                          float* PB, int n_genes, int n_donors, int K)
{
    const int lane = threadIdx.x & 31;
    const int pair = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (pair >= n_genes * n_donors) return;   // uniform across the warp
    const int g = pair / n_donors;
    const int d = pair % n_donors;

    float acc[32];
    #pragma unroll
    for (int k = 0; k < 32; ++k) acc[k] = 0.f;

    for (int i = seg_ptr[pair] + lane; i < seg_ptr[pair + 1]; i += 32) {
        const float v = val[i];
        const unsigned m = mask[col[i]];
        #pragma unroll
        for (int k = 0; k < 32; ++k) acc[k] += ((m >> k) & 1u) ? v : 0.f;
    }

    #pragma unroll
    for (int k = 0; k < 32; ++k) {
        float s = acc[k];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
        if (lane == 0 && k < K) PB[((size_t)d * K + k) * n_genes + g] = s;
    }
}

}  // extern "C"
