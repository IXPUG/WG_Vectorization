#include<immintrin.h>
#include<math.h>
#include<stdlib.h>
#include<stdio.h>
#include<inttypes.h>
__declspec(vector(nomask))
void compute_f(float dij, float dik, float *fi, float *fj, float *fk) {
    *fi = exp(dik) * dij;
    *fj = sin(dij) * dik;
    *fk = - *fi - *fj;
}
__declspec(noinline)
void memory_reduce_add(float *a, float b) {
    *a += b;
}
#ifdef VECTOR_VARIANT
#ifndef __MIC__
__declspec(vector_variant(implements(memory_reduce_add(float* a, float b)), uniform(a), vectorlength(8), nomask))
void memory_reduce_add_avx(float *a, __m128 b1, __m128 b2) {
    __m128 c = _mm_add_ps(b1, b2);
    __m128 cc = _mm_permute_ps(c, 0xB1);
    __m128 d = _mm_add_ps(c, cc);
    __m128 dd = _mm_permute_ps(d, 0x4E);
    *a += _mm_cvtss_f32(_mm_add_ps(d, dd));
}
#else
__declspec(vector_variant(implements(memory_reduce_add(float* a, float b)), uniform(a), vectorlength(16), nomask))
void memory_reduce_add_knc(float *a, __m512 b) {
    *a += _mm512_reduce_add_ps(b);
}
#endif
#endif
__declspec(noinline)
void inner_loop_reduce(int N, int * restrict marr, int * restrict base, int * restrict offs, float * restrict x, float * restrict f) {
//    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        const int M = marr[i];
        const float xi = x[i];
        const int * restrict idxs = base + offs[i];
        float acc_fi = 0.f;
        __assume_aligned(idxs, 64);
        __assume(M % 16 == 0);
#ifdef SIMD
        #pragma simd reduction(+:acc_fi)
#endif
        for (int j = 0; j < M; j++) {
            const int jj = idxs[j];
            const float xj = x[jj];
            const float dxij = xi - xj;
            float acc_fj = 0.f;
            for (int k = 0; k < M; k++) {
                const int kk = idxs[k];
                const float xk = x[kk];
                const float dxik = xi - xk;
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
            f[jj] += acc_fj;
        }
        f[i] += acc_fi;
    }
}
int main(int argc, char **argv) {
    int N = 100;
    int M = 16;
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
        }
    }
    uint64_t start = __rdtsc();
    inner_loop_reduce(N, marr, base, offs, x, f);
    uint64_t end = __rdtsc();
    printf("%21s: Time   : %.15e\n", argv[0] + 20, (float) (end - start));
    double sum = 0;
    for (int i = 0; i < N; i++) {
        sum += f[i];
    }
    printf("%21s: Correct: %.15e\n", argv[0] + 20, (float)sum);
}
