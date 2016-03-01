#include "common.h"

__declspec(noinline)
static void memory_reduce_add(real *dest, real src) {
    *dest += src;
}
void
fill_lanes_naive
(int N, int * iarr, int * jarr, int * marr, int * base, int * offs, 
 real * x, real * f, real rsq) {
    __assume_aligned(iarr, 64);
    __assume_aligned(jarr, 64);
    #pragma simd
    for (int idx = 0; idx < N; idx++) {
        const int i = iarr[idx];
        const int j = jarr[idx];
        const int M = marr[i];
        const real xi = x[i];
        const int * idxs = base + offs[i];
        real acc_fi = 0;
        const real xj = x[j];
        const real dxij = xi - xj;
        real acc_fj = 0;
        for (int k = 0; k < M; k++) {
            const int kk = idxs[k];
            const real xk = x[kk];
            const real dxik = xi - xk;
            if (dxik * dxik > rsq) continue;
            real fi = 0, fj = 0, fk = 0;
            compute_f(dxij, dxik, &fi, &fj, &fk);
            acc_fj += fj;
            acc_fi += fi;
            memory_reduce_add(&f[kk], fk);
        }
        memory_reduce_add(&f[j], acc_fj);
        memory_reduce_add(&f[i], acc_fi);
    }
}
