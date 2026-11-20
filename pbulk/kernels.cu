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


// ===========================================================================
// IR2 variants of pb_gather. Same (gene, donor) segments, same output layout.
// ===========================================================================

// Transposed (recursive-halving) warp reduction. Each lane holds 32 partial sums. At offset o a lane keeps
// half of them and swaps the other half with lane ^ o, so after 5 steps lane k holds the warp's sum for
// accumulator k: 16 + 8 + 4 + 2 + 1 = 31 shuffles instead of 32 x 5 = 160, and every lane writes one output.
// One halving step at a compile-time offset O, so every acc[] index is a constant and the array stays in
// registers. (A first version looped over O at run time. nvcc did not fully unroll the nested loop, the
// array was indexed dynamically and went to local memory: 4.6x slower. See the IR2 report.)
extern "C++" {   // templates need C++ linkage
template <int O>
__device__ __forceinline__ void halve(float acc[32], int lane)
{
    const bool up = (lane & O) != 0;
    #pragma unroll
    for (int j = 0; j < O; ++j) {
        const float send = up ? acc[j] : acc[j + O];
        const float keep = up ? acc[j + O] : acc[j];
        acc[j] = keep + __shfl_xor_sync(0xffffffffu, send, O);
    }
}
}  // extern "C++"

// Transposed (recursive-halving) warp reduction. Each lane holds 32 partial sums. At offset o a lane keeps
// half of them and swaps the other half with lane ^ o, so after 5 steps lane k holds the warp's sum for
// accumulator k: 16 + 8 + 4 + 2 + 1 = 31 shuffles instead of 32 x 5 = 160, and every lane writes one output.
__device__ __forceinline__ float transpose_reduce(float acc[32], int lane)
{
    halve<16>(acc, lane);
    halve<8>(acc, lane);
    halve<4>(acc, lane);
    halve<2>(acc, lane);
    halve<1>(acc, lane);
    return acc[0];
}

// v3 (idea 3a): pb_gather with the transposed reduction. Inputs identical to pb_gather.
__global__ void pb_gather_xr(const int* __restrict__ seg_ptr, const int* __restrict__ col,
                             const float* __restrict__ val, const unsigned* __restrict__ mask,
                             float* PB, int n_genes, int n_donors, int K)
{
    const int lane = threadIdx.x & 31;
    const int pair = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (pair >= n_genes * n_donors) return;
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
    const float s = transpose_reduce(acc, lane);
    if (lane < K) PB[((size_t)d * K + lane) * n_genes + g] = s;
}

// v4 (idea 3): fused mask stream. mval[i] = mask[col[i]] is stored per nonzero at layout time, so the
// loop reads two coalesced streams and no dependent gather. Plus the transposed reduction.
__global__ void pb_gather_fm(const int* __restrict__ seg_ptr, const unsigned* __restrict__ mval,
                             const float* __restrict__ val, float* PB, int n_genes, int n_donors, int K)
{
    const int lane = threadIdx.x & 31;
    const int pair = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (pair >= n_genes * n_donors) return;
    const int g = pair / n_donors;
    const int d = pair % n_donors;

    float acc[32];
    #pragma unroll
    for (int k = 0; k < 32; ++k) acc[k] = 0.f;
    for (int i = seg_ptr[pair] + lane; i < seg_ptr[pair + 1]; i += 32) {
        const float v = val[i];
        const unsigned m = mval[i];
        #pragma unroll
        for (int k = 0; k < 32; ++k) acc[k] += ((m >> k) & 1u) ? v : 0.f;
    }
    const float s = transpose_reduce(acc, lane);
    if (lane < K) PB[((size_t)d * K + lane) * n_genes + g] = s;
}

// v5 (idea 2): complement + set-bit iteration. Pseudo-bulk k of donor d is "flipped" (bit k of flip[d])
// when it takes more than half the donor's cells. mval holds mask ^ flip per nonzero, so its set bits are
// the terms we actually add: the selected cells of unflipped pseudo-bulks and the UNselected cells of
// flipped ones. Each lane walks only its set bits (__ffs) and adds into its own column of a shared-memory
// accumulator. Then PB_k = flipped ? T - s_k : s_k, with T the segment total.
// Work per nonzero: 1 + popcount(mval) adds instead of 32 masked adds.
// Shared accumulator per warp: sacc[k * 33 + lane]. Stride 33 so the final per-k sum (lane k reads its row)
// is bank-conflict free. The scattered adds during the loop can conflict (lanes hit different k).
#define CS_STRIDE 33
__global__ void pb_gather_cs(const int* __restrict__ seg_ptr, const unsigned* __restrict__ mval,
                             const float* __restrict__ val, const unsigned* __restrict__ flip,
                             float* PB, int n_genes, int n_donors, int K)
{
    extern __shared__ float smem[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int pair = blockIdx.x * (blockDim.x >> 5) + warp;
    if (pair >= n_genes * n_donors) return;
    const int g = pair / n_donors;
    const int d = pair % n_donors;
    float* sacc = smem + warp * (32 * CS_STRIDE);

    for (int k = 0; k < 32; ++k) sacc[k * CS_STRIDE + lane] = 0.f;
    __syncwarp();

    float tot = 0.f;
    for (int i = seg_ptr[pair] + lane; i < seg_ptr[pair + 1]; i += 32) {
        const float v = val[i];
        unsigned m = mval[i];
        tot += v;
        while (m) {
            const int k = __ffs(m) - 1;
            m &= m - 1;
            sacc[k * CS_STRIDE + lane] += v;
        }
    }
    __syncwarp();
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);

    float s = 0.f;
    for (int j = 0; j < 32; ++j) s += sacc[lane * CS_STRIDE + j];
    if (lane < K) {
        const bool flipped = (flip[d] >> lane) & 1u;
        PB[((size_t)d * K + lane) * n_genes + g] = flipped ? tot - s : s;
    }
}

}  // extern "C"
