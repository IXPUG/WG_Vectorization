#include<immintrin.h>
#include<math.h>
#include<stdlib.h>
#include<stdio.h>
#include<inttypes.h>
__declspec(vector)
void compute_f(float dij, float dik, float *fi, float *fj, float *fk) {
    *fi = exp(dik) * cos(dij) * sin(dik) * dij;
    *fj = sin(dij) * exp(dij) * cos(dik) * dik;
    *fk = - *fi - *fj;
}
__declspec(noinline)
void memory_reduce_add(float *a, float b) {
    *a += b;
}
__declspec(noinline)
void memory_reduce_lor(int *a, int b) {
    *a = *a || b;
}
__declspec(noinline)
void memory_reduce_land(int *a, int b) {
    *a = *a && b;
}
__declspec(noinline)
void memory_assign(int *a, int b) {
    *a = b;
}
__declspec(noinline)
int reduce_land(int a) {
    return a;
}
__declspec(noinline)
int reduce_lor(int a) {
    return a;
}
#ifdef VECTOR_VARIANT
#ifndef __MIC__
//__declspec(vector_variant(implements(reduce_lor(int a)), vectorlength(8), nomask))
//void reduce_lor_avx(__m128i b1, __m128i b2) {
//    __m128i a = _mm_or_si128(b1, b2);
//    return ! _mm_test_all_zeros(a, a);
//}
//__declspec(vector_variant(implements(reduce_land(int a)), vectorlength(8), nomask))
//void reduce_land_avx(__m128i b1, __m128i b2) {
//    __m128i a = _mm_and_si128(b1, b2);
//    return _mm_test_all_ones(a);
//}
#define lu_full 0xFFFFFFFF
int lu_bool[2][4] = {{0, 0, 0, 0}, {lu_full, lu_full, lu_full, lu_full}}; 
__declspec(vector_variant(implements(reduce_lor(int a)), vectorlength(4), nomask))
__m128i reduce_lor_sse(__m128i b) {
    int a = ! _mm_test_all_zeros(b, b);
    return *(__m128i*)&lu_bool[a];
}
__declspec(vector_variant(implements(reduce_land(int a)), vectorlength(4), nomask))
__m128i reduce_land_sse(__m128i b) {
    return *(__m128i*)&lu_bool[_mm_test_all_ones(b)];
}
#else
//__declspec(vector_variant(implements(reduce_lor(int a)), vectorlength(16), nomask))
//__m512 reduce_lor_knc(__m512 a) {
//    return ! _mm512_kortestz(a, a);
//}
//__declspec(vector_variant(implements(reduce_land(int a)), vectorlength(16), nomask))
//void reduce_land_knc(__mmask16 a) {
//    return _mm512_kortestc(a, a);
//}
#endif
#endif

#ifdef FILL
    int temp;
#endif
__declspec(noinline)
void fill_lanes(int N, int * iarr, int * jarr, int * marr, int * base, int * offs, float * x, float * f, float rsq) {
#ifdef FILL
    int * tempp = &temp;
#endif
#ifdef SIMD
    #pragma simd
#endif
    for (int idx = 0; idx < N; idx++) {
        const int i = iarr[idx];
        const int j = jarr[idx];
        const int M = marr[i];
        const float xi = x[i];
        const int * idxs = base + offs[i];
        float acc_fi = 0.f;
        const float xj = x[j];
        const float dxij = xi - xj;
        float acc_fj = 0.f;
#ifdef FILL
        typedef int bool;
        int k = 0;
        bool active_mask = k < M;
        bool eff_old_mask = 0;
        int kk = idxs[k];
        float xk = x[kk];
  #ifdef FILL_TEMP
        memory_assign(&temp, 0);
        memory_reduce_lor(&temp, active_mask);
        while (temp) {
  #elif defined(FILL_DIRECT)
        while(reduce_lor(active_mask)) {
  //#elif defined(FILL_STORE)
  //      *tempp = 0;
  //      if (active_mask) *tempp = 1;
  //      while (*tempp) {
  #else
        while (/*some*/active_mask) {
  #endif
            bool new_mask = active_mask && ! eff_old_mask;
            if (new_mask) {
                kk = idxs[k];
                xk = x[kk];
            }
            const float dxik = xi - xk;
            bool cutoff_mask = dxik * dxik <= rsq;
            bool eff_mask = cutoff_mask && active_mask;
  #ifdef FILL_TEMP
            memory_assign(&temp, 1);
            memory_reduce_land(&temp, eff_mask || ! active_mask);
            if (temp) {
  #elif defined(FILL_DIRECT)
            if (reduce_land(eff_mask || ! active_mask)) {
  //#elif defined(FILL_STORE)
  //          *tempp = 1;
  //          if (eff_mask || ! active_mask) *tempp = 0;
  //          if (*tempp) {
  #else
            if (/*all*/eff_mask || ! active_mask) {
  #endif
                float fi = 0., fj = 0., fk = 0.;
                compute_f(dxij, dxik, &fi, &fj, &fk);
                acc_fj += fj;
                acc_fi += fi;
  #ifdef SIMD_CORRECT
                memory_reduce_add(&f[kk], fk);
  #else
                f[kk] += fk;
  #endif
                k += 1;
                eff_old_mask = 0;
            } else {
                if (! eff_mask) k += 1;
                eff_old_mask = eff_mask;
            }
            active_mask = active_mask && (k < M);
  #ifdef FILL_TEMP
            memory_assign(&temp, 0);
            memory_reduce_lor(&temp, active_mask);
  //#elif defined(FILL_STORE)
  //          *tempp = 0;
  //          if (active_mask) *tempp = 1;
  #endif
        }
#else
        for (int k = 0; k < M; k++) {
            const int kk = idxs[k];
            const float xk = x[kk];
            const float dxik = xi - xk;
            if (dxik * dxik > rsq) continue;
            float fi = 0., fj = 0., fk = 0.;
            compute_f(dxij, dxik, &fi, &fj, &fk);
            acc_fj += fj;
            acc_fi += fi;
  #ifdef SIMD_CORRECT
            memory_reduce_add(&f[kk], fk);
  #else
            f[kk] += fk;
  #endif
        }
#endif
#ifdef SIMD_CORRECT
        memory_reduce_add(&f[j], acc_fj);
        memory_reduce_add(&f[i], acc_fi);
#else
        f[j] += acc_fj;
        f[i] += acc_fi;
#endif
    }
}
#ifdef FILL_INTR_PHI
__declspec(noinline)
void fill_lanes_intr(int N, int * iarr, int * jarr, int * marr, int * base, int * offs, float * x, float * f, float rsq) {
    float tmpf[16] __attribute__((aligned(64)));
    int tmpi[16] __attribute__((aligned(64)));
    __m512 rsqb = _mm512_set1_ps(rsq);
    __m512i ONE = _mm512_set1_epi32(1);
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
                __m512 fi = _mm512_mul_ps(_mm512_mul_ps(_mm512_exp_ps(dxik), _mm512_cos_ps(dxij)), _mm512_mul_ps(_mm512_sin_ps(dxik), dxij));
                __m512 fj = _mm512_mul_ps(_mm512_mul_ps(_mm512_sin_ps(dxij), _mm512_exp_ps(dxij)), _mm512_mul_ps(_mm512_cos_ps(dxik), dxik));
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
#endif
int main(int argc, char **argv) {
    int N = 10000;
    int M = 16;
    int * iarr = _mm_malloc(N * M * sizeof(*iarr), 64);
    int * jarr = _mm_malloc(N * M * sizeof(*jarr), 64);
    int * marr = _mm_malloc(N * sizeof(*marr), 64);
    int * base = _mm_malloc(N * M * sizeof(*base), 64);
    int * offs = _mm_malloc(N * sizeof(*offs), 64);
    float * x = _mm_malloc(N * sizeof(*x), 64);
    float * f = _mm_malloc(N * sizeof(*f), 64);
    for (int i = 0; i < N; i++) {
        marr[i] = M;
        offs[i] = M * i;
        f[i] = 0.f;
        x[i] = 1.f / N * i;
        for (int j = 0; j < M; j++) {
            base[i * M + j] = (i + j + 1) % N;
            iarr[i * M + j] = i;
            jarr[i * M + j] = j;
        }
    }
    float rsq = 0.25f * M / N;
    rsq *= rsq;
    uint64_t start = __rdtsc();
#ifdef FILL_INTR_PHI
    fill_lanes_intr(N * M, iarr, jarr, marr, base, offs, x, f, rsq);
#else
    fill_lanes(N * M, iarr, jarr, marr, base, offs, x, f, rsq);
#endif
    uint64_t end = __rdtsc();
    printf("%20s: Time   : %.15e\n", argv[0] + 13, (float) (end - start));
    double sum = 0;
    for (int i = 0; i < N; i++) {
        sum += f[i];
    }
    printf("%20s: Correct: %.15e\n", argv[0] + 13, (float)sum);
    printf("%20s: Correct: %.15e\n", argv[0] + 13, (float)f[0]);
}
