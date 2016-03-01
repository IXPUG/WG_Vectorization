#include "common.h"
#include<stdbool.h>

void fill_lanes_chunked(int N, int * iarr, int * jarr, int * marr, int * base, int * offs, real * x, real * f, real rsq) {
    int i[VL], j[VL], M[VL];
    int off[VL], k[VL], kk[VL];
    real xi[VL], xj[VL], xk[VL], dxij[VL], dxik[VL];
    real fi[VL], fj[VL], fk[VL];
    real acc_fi[VL], acc_fj[VL];
    bool active_mask[VL], eff_old_mask[VL], new_mask[VL];
    bool cutoff_mask[VL], eff_mask[VL];
    __assume_aligned(iarr, 64);
    __assume_aligned(jarr, 64);
    for (int idx = 0; idx <= N - VL; idx += VL) {
        __assume(idx % VL == 0);
        #pragma simd
        for (int l = 0; l < VL; l++) {
            i[l] = iarr[idx+l];
            j[l] = jarr[idx+l];
            M[l] = marr[i[l]];
            xi[l] = x[i[l]];
            off[l] = offs[i[l]];
            acc_fi[l] = 0;
            xj[l] = x[j[l]];
            dxij[l] = xi[l] - xj[l];
            acc_fj[l] = 0;
            M[l] += off[l];
            k[l] = off[l];
            active_mask[l] = k[l] < M[l];
            eff_old_mask[l] = false;
            kk[l] = base[k[l]];
            xk[l] = x[kk[l]];
        }
//        i[:] = iarr[idx:VL];
//        j[:] = jarr[idx:VL];
//        M[:] = marr[i[:]];
//        xi[:] = x[i[:]];
//        off[:] = offs[i[:]];
//        acc_fi[:] = 0.0f;
//        xj[:] = x[j[:]];
//        dxij[:] = xi[:] - xj[:];
//        acc_fj[:] = 0.0f;
//        M[:] += off[:];
//
//        k[:] = off[:];
//        active_mask[:] = k[:] < M[:];
//        eff_old_mask[:] = false;
//        kk[:] = base[k[:]];
//        xk[:] = x[kk[:]];
        while (__sec_reduce_any_nonzero(active_mask[:])) {
            new_mask[:] = ! eff_old_mask[:] && active_mask[:];
            if (new_mask[:]) kk[:] = base[k[:]];
            if (new_mask[:]) xk[:] = x[kk[:]];
            dxik[:] = xi[:] - xk[:];
            cutoff_mask[:] = dxik[:] * dxik[:] < rsq;
            eff_mask[:] = cutoff_mask[:] && active_mask[:];
            if (__sec_reduce_all_nonzero(eff_mask[:] || ! active_mask[:])) {
                #pragma simd
                for (int l = 0; l < VL; l++) {
                    compute_f(dxij[l], dxik[l], &fi[l], &fj[l], &fk[l]);
                }
                if (eff_mask[:]) acc_fi[:] += fi[:];
                if (eff_mask[:]) acc_fj[:] += fj[:];
                for (int l = 0; l < VL; l++) {
                    if (eff_mask[l]) {
                        f[kk[l]] += fk[l];
                    }
                }
                k[:] += 1;
                eff_old_mask[:] = 0;
            } else {
                if (! eff_mask[:]) k[:] += 1;
                eff_old_mask[:] = eff_mask[:];
            }
            active_mask[:] &= k[:] < M[:];
        }
        for (int l = 0; l < VL; l++) {
            f[iarr[idx+l]] += acc_fi[l];
        }
        for (int l = 0; l < VL; l++) {
            f[jarr[idx+l]] += acc_fj[l];
        }
    }
}
