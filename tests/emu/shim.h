// Lets kernels.cu compile and run as plain C++ on a machine with no GPU. One OS thread. Blocks and
// warps run one after another. The 32 lanes of a warp are coroutines (ucontext): a lane runs until
// it reaches a warp shuffle, then yields, and the shuffle returns once all 32 lanes have reached it.
// That is the lockstep the kernel relies on. atomicAdd is a plain add because nothing runs concurrently.
// Not modelled: __syncthreads, races between warps, timing.
#pragma once
#ifdef __APPLE__
#define _XOPEN_SOURCE 700   // macOS only declares the full ucontext_t with this. Must come before any system header.
#endif
#include <ucontext.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#define __global__
#define __device__
#define __forceinline__
#define __shared__
#define __restrict__
struct Dim3 { unsigned x = 0, y = 0, z = 0; };
static Dim3 blockIdx, threadIdx, blockDim, gridDim;

namespace emu {
static ucontext_t main_ctx, lane_ctx[32];
static int cur;                       // lane that is running
static bool finished[32], waiting[32];
static float slot[32], res[32];       // value each lane published, value each lane gets back
static int offset[32];
static bool xor_mode[32];         // shuffle kind each lane asked for
}

inline float atomicAdd(float* a, float v) { float o = *a; *a += v; return o; }

inline float __shfl_down_sync(unsigned, float v, int o) {
    const int me = emu::cur;
    emu::slot[me] = v; emu::offset[me] = o; emu::xor_mode[me] = false; emu::waiting[me] = true;
    swapcontext(&emu::lane_ctx[me], &emu::main_ctx);
    return emu::res[me];
}

inline float __shfl_xor_sync(unsigned, float v, int o) {
    const int me = emu::cur;
    emu::slot[me] = v; emu::offset[me] = o; emu::xor_mode[me] = true; emu::waiting[me] = true;
    swapcontext(&emu::lane_ctx[me], &emu::main_ctx);
    return emu::res[me];
}

inline int __ffs(unsigned x) { return __builtin_ffs((int)x); }
inline void __syncwarp() {}   // lanes only meet at shuffles here, and every kernel shuffles before reading shared data

