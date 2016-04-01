#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <random>
#include <malloc.h>
#include <x86intrin.h>

#include "timer.h"

void _mm_print(const __m256d x, const uint8_t mask, size_t i, const char *str)
{
#ifdef VERBOSE
    double *mm = (double*)&x;
    fprintf(stderr, "%s [%4lu] [% .2f % .2f % .2f % .2f] [%x] (%c%c%c%c)\n",
            str, i, mm[3], mm[2], mm[1], mm[0], mask,
            mask & 8 ? '1' : '0', mask & 4 ? '1' : '0',
            mask & 2 ? '1' : '0', mask & 1 ? '1' : '0');
#endif
}

/*
 * Function to sort SIMD vector based on mask, active lanes are marked as 1 in
 * mask, and are moved towards lower part of the vector. The variable shtab is
 * a true/false table of which shuffles to perform on mm in order to sort active
 * and inactive lanes.
 */

void _mm_sort_pd(__m256d& mm, const uint8_t& mask)
{
    const uint64_t shtab = 0x311269308410200;
    uint8_t code = (uint8_t)((shtab >> 4*mask) & 0x0f);
    if (code == 0) return;
    if (code & 1) mm = _mm256_permute4x64_pd(mm, 0x4e);
    if (code & 2) mm = _mm256_permute4x64_pd(mm, 0xb1);
    if (code & 4) mm = _mm256_permute4x64_pd(mm, 0xd8);
    if (code & 8) mm = _mm256_permute4x64_pd(mm, 0x39);
}

/*
 * Base test is: apply f(x) for numbers with mask == 0, and g(x) to numbers for
 * which mask == 1 in the mask vector, and verify if there is an advantage in
 * performance to reorder the data. The function below simply does a pass on the
 * data assigning f(x)/g(x) using a mask.
 */

void _mm_apply_func_pd(__m256d *mm, uint8_t *mask, size_t n,
                       __m256d (*f)(__m256d), __m256d (*g)(__m256d))
{
    for (size_t i = 0; i < n; i++) {
        __m256d m256mask = _mm256_set_pd(mask[i] & 8, mask[i] & 4, mask[i] & 2, mask[i] & 1);
        mm[i] = _mm256_blendv_pd(f(mm[i]), g(mm[i]), m256mask);
    }
}

/*
 * Same as function above, but using streaming loads/stores, since it does
 * a single pass through the data. In another context, it might make sense
 * to not bypass the caches.
 */

void _mm_apply_func_stream_pd(__m256d *mm, uint8_t *mask, size_t n,
                       __m256d (*f)(__m256d), __m256d (*g)(__m256d))
{
    for (size_t i = 0; i < n; i++) {
        __m256d mmtmp = (__m256d) _mm256_stream_load_si256((__m256i*)&mm[i]);
        __m256d m256mask = _mm256_set_pd(mask[i] & 8, mask[i] & 4, mask[i] & 2, mask[i] & 1);
        mmtmp = _mm256_blendv_pd(f(mmtmp), g(mmtmp), m256mask);
        _mm256_stream_pd((double*)&mm[i], mmtmp);
    }
}

/*
 * Additionally check when mask is all true or all false to avoid computing f(x)/g(x) when not
 * needed.
 */

void _mm_apply_func_branch_pd(__m256d *mm, uint8_t *mask, size_t n,
                       __m256d (*f)(__m256d), __m256d (*g)(__m256d))
{
    for (size_t i = 0; i < n; i++) {
        __m256d mmtmp = (__m256d) _mm256_stream_load_si256((__m256i*)&mm[i]);

        if (mask[i] == 0x0) {
            _mm256_stream_pd((double*)&mm[i], f(mmtmp));
        } else if (mask[i] == 0xf) {
            _mm256_stream_pd((double*)&mm[i], g(mmtmp));
        } else {
            __m256d m256mask = _mm256_set_pd(mask[i] & 8, mask[i] & 4, mask[i] & 2, mask[i] & 1);
            mmtmp = _mm256_blendv_pd(f(mmtmp), g(mmtmp), m256mask);
            _mm256_stream_pd((double*)&mm[i], mmtmp);
        }
    }
}

/*
 * SIMD lane sorting technique. Sort data into groups of elements where
 * mask == true or mask == false, and only calls f(x) or g(x) for each group after
 * sorted. This always avoids calling f(x) or g(x) if it doesn't apply to that
 * piece of data. The basic idea is to push groups with mixed true/false elements
 * into a set of stacks based on how many lanes are active, then pop stacks with
 * 4 elements into 4 sorted vectors that have all active or all inactive lanes.
 */

void _mm_sort_and_apply_func_pd(__m256d *mm, uint8_t *mask, size_t n,
                                __m256d (*f)(__m256d), __m256d (*g)(__m256d))
{
    size_t i, out = 0;
    static const uint32_t count = 0x29949440;
    __m256d stack[3][4]; uint8_t smask[3][4], index[3] = {0, };

    for (i = 0; i < n; i++) {
        __m256d mmtmp = (__m256d) _mm256_stream_load_si256((__m256i*)&mm[i]);

        /* if mask is full or empty, no sorting is needed, just move on */

        if (mask[i] == 0) {
            mask[out] = mask[i];
            _mm_print(mmtmp, mask[i], i, "empty:");
            _mm256_stream_pd((double*)&mm[out++], f(mmtmp));
            continue;
        } else if (mask[i] == 0xf) {
            mask[out] = mask[i];
            _mm_print(mmtmp, mask[i], i, "full: ");
            _mm256_stream_pd((double*)&mm[out++], g(mmtmp));
            continue;
        }

        /* vector has 1-3 active lanes, count lanes and push to right stack. */

        uint8_t idx = (count >> 2*mask[i]) & 0x03;
        smask[idx][index[idx]] = mask[i];
        stack[idx][index[idx]++] = mmtmp;

#ifdef VERBOSE
        fprintf(stderr, "push into stack %d: ", idx+1);
        _mm_print(mmtmp, mask[i], idx, "");
#endif

        /* if a stack becomes full, sort vectors and transpose */

        if (index[idx] == 4) {
#ifdef VERBOSE
            fprintf(stderr, "pop stack %d\n", idx+1);
#endif
            switch (idx) {
                case 1: { /* stack with 2 lanes active */
                    index[1] = 0;
                    _mm_sort_pd(stack[1][0],  smask[1][0]);
                    _mm_sort_pd(stack[1][1], ~smask[1][1]);
                    _mm_sort_pd(stack[1][2],  smask[1][2]);
                    _mm_sort_pd(stack[1][3], ~smask[1][3]);
                    mask[out] = 0x0; mask[out+1] = 0x0; mask[out+2] = 0xf; mask[out+3] = 0xf;
                    _mm_print(_mm256_blend_pd(stack[1][0], stack[1][1], 0x3), mask[out+0], out+0, "1: ");
                    _mm_print(_mm256_blend_pd(stack[1][2], stack[1][3], 0x3), mask[out+1], out+1, "2: ");
                    _mm_print(_mm256_blend_pd(stack[1][0], stack[1][1], 0xc), mask[out+2], out+2, "3: ");
                    _mm_print(_mm256_blend_pd(stack[1][2], stack[1][3], 0xc), mask[out+3], out+3, "4: ");
                    _mm256_stream_pd((double*)&mm[out++], f(_mm256_blend_pd(stack[1][0], stack[1][1], 0x3)));
                    _mm256_stream_pd((double*)&mm[out++], f(_mm256_blend_pd(stack[1][2], stack[1][3], 0x3)));
                    _mm256_stream_pd((double*)&mm[out++], g(_mm256_blend_pd(stack[1][0], stack[1][1], 0xc)));
                    _mm256_stream_pd((double*)&mm[out++], g(_mm256_blend_pd(stack[1][2], stack[1][3], 0xc)));
                } break;

                case 0: { /* stack with 1 lane active */
                    index[0] = 0;
                    _mm_sort_pd(stack[0][0],  smask[0][0]);
                    _mm_sort_pd(stack[0][1],  smask[0][1]);
                    _mm_sort_pd(stack[0][2], ~smask[0][2]);
                    _mm_sort_pd(stack[0][3], ~smask[0][3]);
                    mask[out] = 0x0; mask[out+1] = 0x0; mask[out+2] = 0x0; mask[out+3] = 0xf;
                    __m256d tmp0 = _mm256_unpackhi_pd(stack[0][0], stack[0][1]);
                    __m256d tmp1 = _mm256_unpacklo_pd(stack[0][2], stack[0][3]);
                    __m256d tmp2 = _mm256_unpacklo_pd(stack[0][0], stack[0][1]);
                    __m256d tmp3 = _mm256_unpackhi_pd(stack[0][2], stack[0][3]);
                    _mm_print(tmp0, mask[out+0], i, "1: ");
                    _mm_print(tmp1, mask[out+1], i, "2: ");
                    _mm_print(_mm256_blend_pd(tmp2, tmp3, 0x3), mask[out+2], i, "3: ");
                    _mm_print(_mm256_blend_pd(tmp2, tmp3, 0xc), mask[out+3], i, "4: ");
                    _mm256_stream_pd((double*)&mm[out++], f(tmp0));
                    _mm256_stream_pd((double*)&mm[out++], f(tmp1));
                    _mm256_stream_pd((double*)&mm[out++], f(_mm256_blend_pd(tmp2, tmp3, 0x3)));
                    _mm256_stream_pd((double*)&mm[out++], g(_mm256_blend_pd(tmp2, tmp3, 0xc)));
                } break;

                case 2: { /* stack with 3 lanes active */
                    index[2] = 0;
                    _mm_sort_pd(stack[2][0],  smask[2][0]);
                    _mm_sort_pd(stack[2][1],  smask[2][1]);
                    _mm_sort_pd(stack[2][2], ~smask[2][2]);
                    _mm_sort_pd(stack[2][3], ~smask[2][3]);
                    mask[out] = 0x0; mask[out+1] = 0xf; mask[out+2] = 0xf; mask[out+3] = 0xf;
                    __m256d tmp0 = _mm256_unpackhi_pd(stack[2][0], stack[2][1]);
                    __m256d tmp1 = _mm256_unpacklo_pd(stack[2][2], stack[2][3]);
                    __m256d tmp2 = _mm256_unpacklo_pd(stack[2][0], stack[2][1]);
                    __m256d tmp3 = _mm256_unpackhi_pd(stack[2][2], stack[2][3]);
                    _mm_print(_mm256_blend_pd(tmp0, tmp1, 0x3), mask[out+0], i, "1: ");
                    _mm_print(_mm256_blend_pd(tmp0, tmp1, 0xc), mask[out+1], i, "2: ");
                    _mm_print(tmp2, mask[out+2], i, "3: ");
                    _mm_print(tmp3, mask[out+3], i, "4: ");
                    _mm256_stream_pd((double*)&mm[out++], f(_mm256_blend_pd(tmp0, tmp1, 0x3)));
                    _mm256_stream_pd((double*)&mm[out++], g(_mm256_blend_pd(tmp0, tmp1, 0xc)));
                    _mm256_stream_pd((double*)&mm[out++], g(tmp2));
                    _mm256_stream_pd((double*)&mm[out++], g(tmp3));
                } break;
            }
        }
    }

    /* empty stacks into output vector */

    for (out = n, i = 0; i < 3; i++) {
#ifdef VERBOSE
        if (index[i] > 0)
            fprintf(stderr, "empty stack %lu (%d)\n", i, index[i]);
#endif
        while (index[i] > 0) {
            _mm_print(stack[i][index[i]-1], smask[i][index[i]-1], i, "  ");
            mm[--out] = stack[i][--index[i]];
            mask[out] = smask[i][index[i]];
        }
    }

    /*
     * use regular masks to apply f(x)/g(x) to remainder (up to 9 elements if
     * all stacks are almost full). An alternative is to combine stack elements
     * from stacks 1 and 3, and from stack 2 with blending, to leave at most one
     * element ungrouped.
     */

    _mm_apply_func_pd(&mm[out], &mask[out], n-out, f, g);
}

/* callbacks to compute when mask == false and mask == true */

__m256d inv(__m256d x)
{
    return _mm256_div_pd(_mm256_set1_pd(1.0), x);
}

__m256d f(__m256d x)
{
    return inv(inv(x));
}

__m256d g(__m256d x)
{
    return f(_mm256_sqrt_pd(_mm256_mul_pd(x,x)));
}

/* Main function generates random numbers and applies f/g according to masks */

int main(int argc, char *argv[])
{
    size_t i, n = (size_t) strtol(argv[1], NULL, 10);

    double a = strtod(argv[2], NULL);
    double b = strtod(argv[3], NULL);
    double c = strtod(argv[4], NULL);

    uint8_t *mask = (uint8_t*) malloc(n * sizeof(uint8_t));
    __m256d *mm = (__m256d*) memalign(32, n * sizeof(__m256d));

    /* create random numbers in interval (a,b) */

    std::default_random_engine rng;
    std::uniform_real_distribution<double> dist(a, b);
    auto uniform_ab = std::bind(dist, rng);

    for (i = 0; i < n; i++) {
        double x, y, z, w;

        x = uniform_ab();
        y = uniform_ab();
        z = uniform_ab();
        w = uniform_ab();

        mm[i] = _mm256_set_pd(w, z, y, x);
        mask[i] = (w > c) << 3 | (z > c) << 2 | (y > c) << 1 | (x > c);
    }

    /* print input */

#ifdef VERBOSE
    fprintf(stderr, "input:\n");
    for (i = 0; i < n; i++) {
        double *mmptr = (double*)&mm[i];
        fprintf(stderr, "[%4lu] [% .2f % .2f % .2f % .2f] [%x] (%c%c%c%c)\n",
                i, mmptr[3], mmptr[2], mmptr[1], mmptr[0], mask[i],
                mask[i] & 8 ? '1' : '0', mask[i] & 4 ? '1' : '0',
                mask[i] & 2 ? '1' : '0', mask[i] & 1 ? '1' : '0');
    }
#endif

    /* apply x = f(x) if x <= c, else x = g(x) */

    Timer<> timer;

#ifdef SORT_SIMD_LANES
    _mm_sort_and_apply_func_pd(mm, mask, n, f, g);
#else
    /* use best masking implementation */
    _mm_apply_func_branch_pd(mm, mask, n, f, g);
#endif

    double total_time = timer.elapsed();

#ifdef VERBOSE
    fprintf(stderr, "\noutput:\n");

    for (i = 0; i < n; i++) {
        double *mmptr = (double*)&mm[i];
        fprintf(stderr, "[%4lu] [% .2f % .2f % .2f % .2f] (%c%c%c%c)\n",
                i, mmptr[3], mmptr[2], mmptr[1], mmptr[0],
                mask[i] & 8 ? '1' : '0', mask[i] & 4 ? '1' : '0',
                mask[i] & 2 ? '1' : '0', mask[i] & 1 ? '1' : '0');
    }
#endif
    fprintf(stderr, "elapsed time: %.4f s\n", 1e-9 * total_time);

    return 0;
}
