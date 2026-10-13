// Sequential C++ reference, one thread. This is Algorithm 1 of the report written out directly:
// for every pseudo-bulk, for every selected cell, add the cell's nonzeros into the pseudo-bulk's row,
// then scale the row to sum to 1e6. Same inputs as pb_scatter (S as CSR, X as CSC), float32 like the GPU.
// Built as a shared library and called through ctypes (see cpu_cpp.py).
#include <cstddef>

extern "C" void pb_sequential(const int* sel_ptr, const int* sel_idx, const int* col_ptr,
                              const int* row_idx, const float* val, float* PB, int n_pb, int n_genes)
{
    for (int p = 0; p < n_pb; ++p) {
        float* out = PB + (size_t)p * n_genes;
        for (int g = 0; g < n_genes; ++g) out[g] = 0.f;
        for (int i = sel_ptr[p]; i < sel_ptr[p + 1]; ++i) {
            const int c = sel_idx[i];
            for (int j = col_ptr[c]; j < col_ptr[c + 1]; ++j) out[row_idx[j]] += val[j];
        }
        double total = 0.0;
        for (int g = 0; g < n_genes; ++g) total += out[g];
        const float scale = (float)(1e6 / total);
        for (int g = 0; g < n_genes; ++g) out[g] *= scale;
    }
}
