#include "common.h"
#include <math.h>
#include<stdlib.h>
#define ALIGNMT 64

#ifdef __INTEL_COMPILER
    #define ALIGNED( x ) __assume_aligned( (x), ALIGNMT )
    #define ALIGN __attribute__(( aligned( ALIGNMT ) ))
#elif defined( __GNUC__ )
    #define ALIGNED( x )
    #define ALIGN __attribute__(( aligned( ALIGNMT ) ))
#else
    #define ALIGNED( x )
    #define ALIGN
#endif

struct linearised_data {
  int N; 
  int M;
  int * linIdx;
};

void * linearised_init(int N, int M) {
  struct linearised_data * ret = malloc(sizeof(struct linearised_data));
  ret->N = N;
  ret->M = M;
  ret->linIdx = _mm_malloc(N * M * M * sizeof(int), 64);
  return ret;
}

void linearised_finish(void * data) {
  struct linearised_data * d = data;
  _mm_free(d->linIdx);
  free(d);
}


int fill_lanes_linearised(int N, int * iarr, int * jarr, int * marr, int * base, int * offs,
                float * x, float *ff, float rsq, void * data) {
    struct linearised_data * d = data;
    const int NN = d->N;
    const int MM = d->M;
    int * linIdx = d->linIdx;
    int len = 0;
    for (int idx = 0; idx < N; idx++) {
        const int i = iarr[idx];
        const float xi = x[i];
        const int M = marr[i];
        const int * idxs = base + offs[i];
        for (int k = 0; k < M; k++) {
            const int kk = idxs[k];
            const float xk = x[kk];
            const float dxik = xi - xk;
            if (dxik * dxik <= rsq) {
                linIdx[len++] = idx*MM+k;
            }
        }
    }
    #pragma omp simd aligned( ff, linIdx, iarr, jarr, marr, x, base, offs: ALIGNMT)
    for (int nIdx = 0; nIdx < len; nIdx++) {
        // as ff is aligned and N * VLEN * sizeof(float) long, f is aligned as well
        ALIGN float *f=&ff[(nIdx%VL)*NN]; ALIGNED(f); // one array of size NN per vector lane

        const int idx = linIdx[nIdx] / MM;
        const int k = linIdx[nIdx] % MM;
        const int i = iarr[idx];
        const int j = jarr[idx];
        const int M = marr[i];
        const float xi = x[i];
        const int * idxs = base + offs[i];
        const float xj = x[j];
        const float dxij = xi - xj;
        const int kk = idxs[k];
        const float xk = x[kk];
        const float dxik = xi - xk;
        float fi, fj, fk;
        compute_f(dxij, dxik, &fi, &fj, &fk);
        f[kk] += fk;
        f[j] += fj;
        f[i] += fi;
    }
    #pragma omp simd aligned(ff: ALIGNMT)
    for (int i = 0; i < NN; i++) {
        for (int j = 1; j < VL; j++) {
            ff[i] += ff[j*NN+i];
        }
    }
}

