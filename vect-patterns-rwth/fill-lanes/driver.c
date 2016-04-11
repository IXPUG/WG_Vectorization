#include "common.h"
#include<stdlib.h>
#include<stdio.h>
#include<inttypes.h>

void fill_lanes_chunked(int N, int * iarr, int * jarr, int * marr, int * base, int * offs, real * x, real * f, real rsq, void * data);
void fill_lanes_collect_compute(int N, int * iarr, int * jarr, int * marr, int * base, int * offs, real * x, real * f, real rsq, void * data);
void fill_lanes_naive(int N, int * iarr, int * jarr, int * marr, int * base, int * offs, real * x, real * f, real rsq, void * data);
void fill_lanes_scalar(int N, int * iarr, int * jarr, int * marr, int * base, int * offs, real * x, real * f, real rsq, void * data);
void fill_lanes_intr(int N, int * iarr, int * jarr, int * marr, int * base, int * offs, float * x, float * f, float rsq, void * data);

void fill_lanes_linearised(int N, int * iarr, int * jarr, int * marr, int * base, int * offs, float * x, float * f, float rsq, void * data);
void * linearised_init(int N, int M);
void linearised_finish(void * data);

typedef void (*fill_lanes_t)(int N, int * iarr, int * jarr, int * marr, int * base, int * offs, real * x, real * f, real rsq, void * data);
typedef void* (*fill_lanes_init_t)(int N, int M);

struct impl {
    char * name;
    fill_lanes_t fn;
    fill_lanes_init_t init_fn;
    void (*finish_fn)(void *);
};

struct impl impls[] = {
    {"scalar", fill_lanes_scalar, NULL, NULL},
    {"naive", fill_lanes_naive, NULL, NULL},
    {"collect_compute", fill_lanes_collect_compute, NULL, NULL},
    {"chunked", fill_lanes_chunked, NULL, NULL},
    {"linearised", fill_lanes_linearised, linearised_init, linearised_finish},
#if defined(__MIC__) && defined(MODE_SINGLE)
    {"intr", fill_lanes_intr, NULL, NULL},
#endif
    {NULL, NULL, NULL, NULL}};

int main(int argc, char **argv) {
    int N = 10000;
    int M = 16;
    int * iarr = _mm_malloc(N * M * sizeof(*iarr), 64);
    int * jarr = _mm_malloc(N * M * sizeof(*jarr), 64);
    int * marr = _mm_malloc(N * sizeof(*marr), 64);
    int * base = _mm_malloc(N * M * sizeof(*base), 64);
    int * offs = _mm_malloc(N * sizeof(*offs), 64);
    real * x = _mm_malloc(N * sizeof(*x), 64);
    real * f = _mm_malloc(N * VL * sizeof(*f), 64);
    for (int i = 0; i < N; i++) {
        marr[i] = M;
        offs[i] = M * i;
        f[i] = 0;
        x[i] = 1.0 / N * i;
        for (int j = 0; j < M; j++) {
            base[i * M + j] = (i + j + 1) % N;
            iarr[i * M + j] = i;
            jarr[i * M + j] = (j + 1) % N;
        }
        for (int j = 0; j < M - 1; j++) {
            int k = rand() % (M - j);
            int tmp = base[i * M + j];
            base[i * M + j] = base[i * M + j + k];
            base[i * M + j + k] = tmp;
        }
    }
    real rsq = 0.251 * M / N;
    rsq *= rsq;
    struct impl * cur_impl = impls;
    while (cur_impl->name) {
        void * data = 0;
        if (cur_impl->init_fn) data = cur_impl->init_fn(N, M);
        uint64_t min_t = UINT64_MAX;
        for (int k = 0; k < 100; k++) {
          for (int i = 0; i < N*VL; i++) f[i] = 0;
          uint64_t start = __rdtsc();
          cur_impl->fn(N * M, iarr, jarr, marr, base, offs, x, f, rsq, data);
          uint64_t end = __rdtsc();
          uint64_t t = end - start;
          if (t < min_t) min_t = t;
        }
        printf("%20s: time %.5e f[0] %.5e f[16] %.5e\n", cur_impl->name, (float) (min_t), (float) f[0], (float) f[16]);
        if (cur_impl->finish_fn) cur_impl->finish_fn(data);
        cur_impl++;
    }
}
