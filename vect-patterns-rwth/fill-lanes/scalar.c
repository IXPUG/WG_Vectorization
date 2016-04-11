#include "common.h"

void
fill_lanes_scalar
(int N, int * iarr, int * jarr, int * marr, int * base, int * offs, 
 real * x, real * f, real rsq, void * data) {
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
            f[kk] += fk;
        }
        f[i] += acc_fi;
        f[j] += acc_fj;
    }
}
