#include<immintrin.h>
__declspec(noinline)
void fill_lanes_intr(int N, int * iarr, int * jarr, int * marr, int * base, int * offs, float * x, float * f, float rsq) {
    float tmpf[16] __attribute__((aligned(64)));
    int tmpi[16] __attribute__((aligned(64)));
    __m512 rsqb = _mm512_set1_ps(rsq);
    __m512i ONE = _mm512_set1_epi32(1);
    __m512 TEN = _mm512_set1_ps(10);
    for (int idx = 0; idx < N - 15; idx += 16) {
        __m512i i = _mm512_load_epi32(&iarr[idx]);
        __m512i j = _mm512_load_epi32(&jarr[idx]);
        __m512i M = _mm512_i32gather_epi32(i, marr, 4);
        __m512 xi = _mm512_i32gather_ps(i, x, 4);
//        const int * idxs = base + offs[i];
        __m512i off = _mm512_i32gather_epi32(i, offs, 4);
        __m512 acc_fi = _mm512_setzero_ps();
        __m512 xj = _mm512_i32gather_ps(j, x, 4);
        __m512 dxij = _mm512_sub_ps(xi, xj);
        __m512 acc_fj = _mm512_setzero_ps();
        M = _mm512_add_epi32(M, off);

        __m512i k = off;
        __mmask16 active_mask = _mm512_cmplt_epi32_mask(k, M);
        __mmask16 eff_old_mask = 0;
        __m512i kk = _mm512_i32gather_epi32(k, base, 4);
        __m512 xk = _mm512_i32gather_ps(kk, x, 4);
        while (/*some*/active_mask != 0) {
            __mmask16 new_mask = _mm512_kandn(eff_old_mask, active_mask);
            kk = _mm512_mask_i32gather_epi32(kk, new_mask, k, base, 4);
            xk = _mm512_mask_i32gather_ps(xk, new_mask, kk, x, 4);
            __m512 dxik = _mm512_sub_ps(xi, xk);
            __mmask16 cutoff_mask = _mm512_cmple_ps_mask(_mm512_mul_ps(dxik, dxik), rsqb);
            __mmask16 eff_mask = _mm512_kand(cutoff_mask, active_mask);
            if (_mm512_kortestc(eff_mask, _mm512_knot(active_mask))) {
                __m512 dxij10 = _mm512_mul_ps(TEN, dxij);
                __m512 dxik10 = _mm512_mul_ps(TEN, dxik);
                __m512 fi = _mm512_mul_ps(_mm512_mul_ps(_mm512_exp_ps(dxik10), _mm512_cos_ps(dxij10)), _mm512_mul_ps(_mm512_sin_ps(dxik10), dxij10));
                __m512 fj = _mm512_mul_ps(_mm512_mul_ps(_mm512_sin_ps(dxij10), _mm512_exp_ps(dxij10)), _mm512_mul_ps(_mm512_cos_ps(dxik10), dxik10));
                __m512 fk = _mm512_sub_ps(_mm512_setzero_ps(), _mm512_add_ps(fi, fj));
                acc_fi = _mm512_mask_add_ps(acc_fi, eff_mask, acc_fi, fi);
                acc_fj = _mm512_mask_add_ps(acc_fj, eff_mask, acc_fj, fj);
                _mm512_store_ps(tmpf, fk);
                _mm512_store_epi32(tmpi, kk);
                for (int l = 0; l < 16; l++) {
                    if (eff_mask & (1 << l)) {
                        f[tmpi[l]] += tmpf[l];
                    }
                }
                k = _mm512_add_epi32(k, ONE);
                eff_old_mask = 0;
            } else {
                k = _mm512_mask_add_epi32(k, _mm512_knot(eff_mask), k, ONE);
                eff_old_mask = eff_mask;
            }
            active_mask = _mm512_kand(active_mask, _mm512_cmplt_epi32_mask(k, M));
        }
        _mm512_store_ps(tmpf, acc_fi);
        for (int l = 0; l < 16; l++) {
            f[iarr[idx+l]] += tmpf[l];
        }
        _mm512_store_ps(tmpf, acc_fj);
        for (int l = 0; l < 16; l++) {
            f[jarr[idx+l]] += tmpf[l];
        }
    }
}
