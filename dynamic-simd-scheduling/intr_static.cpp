/**********************************************************
 *
 * Copyright (c) 2016 
 * Olaf Krzikalla, olaf.krzikalla@tu-dresden.de
 * Florian Wende, wende@zib.de 
 * Markus Höhnerbach, hoehnerbach@aices.rwth-aachen.de
 *
 * The intr_static approach:
 * Vectorization approach: intrinsics
 * Scheduling approach: static
 *
 **********************************************************/

#include "main.h"
#include <immintrin.h> 

#if defined(__MIC__) || defined(__AVX512F__)
#ifdef __MIC__
#define _mm512_my_cvtepi32_ps(a)                                               \
  (_mm512_cvtfxpnt_round_adjustepi32_ps((a), _MM_FROUND_CUR_DIRECTION,         \
                                        _MM_EXPADJ_NONE))
#else
#define _mm512_my_cvtepi32_ps(a) (_mm512_cvtepi32_ps(a))
#endif 
#endif

//-----------------------------------------------------------------------------
void Mandelbrot(float x1, float y1, float x2, float y2, int width, int height, int maxIters, int * image)
{
  float dx = (x2-x1)/width, dy = (y2-y1)/height;

  __m512i asc =
      _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
  __m512 vx1 = _mm512_set1_ps(x1);
  __m512 vdx = _mm512_set1_ps(dx);
  __m512 boundary = _mm512_set1_ps(4.0f);
  __m512i vmaxIters = _mm512_set1_epi32(maxIters);
  __m512i vwidth = _mm512_set1_epi32(width);
  __m512i v1 = _mm512_set1_epi32(1);
  __m512i v16 = _mm512_set1_epi32(16);

#ifdef WITH_OMP
#pragma omp parallel for schedule(dynamic, 2)
#endif
  for (int j = 0; j < height; j += 1) {
    __m512 im_c = _mm512_set1_ps(y1 + dy * j);
    int *image_j = image + j * width;
    for (int ii = 0; ii < width; ii += 16) {
      __m512i i = _mm512_add_epi32(_mm512_set1_epi32(ii), asc);
      __m512 re_c =
          _mm512_add_ps(vx1, _mm512_mul_ps(vdx, _mm512_my_cvtepi32_ps(i)));
      __m512 re_z = _mm512_setzero_ps();
      __m512 im_z = _mm512_setzero_ps();
      __m512i k = _mm512_setzero_epi32();
      __m512 re_z2 = _mm512_mul_ps(re_z, re_z);
      __m512 im_z2 = _mm512_mul_ps(im_z, im_z);
      int across = 1;
      for (int kk = 0; kk < maxIters && across; kk++) {
        __m512 norm2 = _mm512_add_ps(re_z2, im_z2);
        __mmask16 end = _mm512_cmplt_ps_mask(norm2, boundary);
        across = !_mm512_kortestz(end, end);
        k = _mm512_mask_add_epi32(k, end, k, v1);
        im_z = _mm512_add_ps(_mm512_mul_ps(_mm512_add_ps(re_z, re_z), im_z),
                             im_c);
        re_z = _mm512_add_ps(re_z2, _mm512_sub_ps(re_c, im_z2));
        re_z2 = _mm512_mul_ps(re_z, re_z);
        im_z2 = _mm512_mul_ps(im_z, im_z);
      }
      _mm512_i32scatter_epi32(image_j, i, k, 4);
    }
  } 
}

