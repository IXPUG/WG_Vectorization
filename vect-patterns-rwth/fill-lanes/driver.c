#include "common.h"
#include<stdlib.h>
#include<stdio.h>
#include<inttypes.h>

void fill_lanes_chunked(int N, int * iarr, int * jarr, int * marr, int * base, int * offs, real * x, real * f, real rsq);
void fill_lanes_collect_compute(int N, int * iarr, int * jarr, int * marr, int * base, int * offs, real * x, real * f, real rsq);
void fill_lanes_naive(int N, int * iarr, int * jarr, int * marr, int * base, int * offs, real * x, real * f, real rsq);
void fill_lanes_scalar(int N, int * iarr, int * jarr, int * marr, int * base, int * offs, real * x, real * f, real rsq);
void fill_lanes_intr(int N, int * iarr, int * jarr, int * marr, int * base, int * offs, float * x, float * f, float rsq);

typedef void (*fill_lanes_t)(int N, int * iarr, int * jarr, int * marr, int * base, int * offs, real * x, real * f, real rsq);

struct impl {
    char * name;
    fill_lanes_t fn;
};

struct impl impls[] = {
    {"scalar", fill_lanes_scalar},
    {"naive", fill_lanes_naive},
    {"collect_compute", fill_lanes_collect_compute},
    {"chunked", fill_lanes_chunked},
#if defined(__MIC__) && defined(MODE_SINGLE)
    {"intr", fill_lanes_intr},
#endif
    {NULL, NULL}};

int main(int argc, char **argv) {
    int N = 32;
    int M = 16;
    int * iarr = _mm_malloc(N * M * sizeof(*iarr), 64);
    int * jarr = _mm_malloc(N * M * sizeof(*jarr), 64);
    int * marr = _mm_malloc(N * sizeof(*marr), 64);
    int * base = _mm_malloc(N * M * sizeof(*base), 64);
    int * offs = _mm_malloc(N * sizeof(*offs), 64);
    real * x = _mm_malloc(N * sizeof(*x), 64);
    real * f = _mm_malloc(N * sizeof(*f), 64);
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
        uint64_t start = __rdtsc();
        cur_impl->fn(N * M, iarr, jarr, marr, base, offs, x, f, rsq);
        uint64_t end = __rdtsc();
        printf("%20s: time %.5e f[0] %.5e f[16] %.5e\n", cur_impl->name, (float) (end - start), (float) f[0], (float) f[16]);
        for (int i = 0; i < N; i++) f[i] = 0;
        cur_impl++;
    }
}
