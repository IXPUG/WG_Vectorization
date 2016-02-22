#include<math.h>
#include<stdlib.h>
#include<stdio.h>
#include<inttypes.h>
void compute_f(float dij, float dik, float *fi, float *fj, float *fk) {
    *fi = exp(dik) * cos(dij) * sin(dik) * dij;
    *fj = sin(dij) * exp(dij) * cos(dik) * dik;
    *fk = - *fi - *fj;
}
void fill_lanes(int N, int * iarr, int * jarr, int * marr, int * base, int * offs, float * x, float * f, float rsq) {
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
        for (int k = 0; k < M; k++) {
            const int kk = idxs[k];
            const float xk = x[kk];
            const float dxik = xi - xk;
            if (dxik * dxik > rsq) continue;
            float fi = 0., fj = 0., fk = 0.;
            compute_f(dxij, dxik, &fi, &fj, &fk);
            acc_fj += fj;
            acc_fi += fi;
            f[kk] += fk;
        }
        f[j] += acc_fj;
        f[i] += acc_fi;
    }
}
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
    fill_lanes(N * M, iarr, jarr, marr, base, offs, x, f, rsq);
    uint64_t end = __rdtsc();
    printf("%20s: Time   : %.15e\n", argv[0] + 13, (float) (end - start));
    double sum = 0;
    for (int i = 0; i < N; i++) {
        sum += f[i];
    }
    printf("%20s: Correct: %.15e\n", argv[0] + 13, (float)sum);
    printf("%20s: Correct: %.15e\n", argv[0] + 13, (float)f[0]);
}
