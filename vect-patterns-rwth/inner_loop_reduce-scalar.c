#include<math.h>
#include<stdlib.h>
#include<stdio.h>
#include<inttypes.h>
void compute_f(float dij, float dik, float *fi, float *fj, float *fk) {
    *fi = exp(dik) * dij;
    *fj = sin(dij) * dik;
    *fk = - *fi - *fj;
}
void inner_loop_reduce(int N, int * restrict marr, int * restrict base, int * restrict offs, float * restrict x, float * restrict f) {
    for (int i = 0; i < N; i++) {
        const int M = marr[i];
        const float xi = x[i];
        const int * restrict idxs = base + offs[i];
        float acc_fi = 0.f;
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
                f[kk] += fk;
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
