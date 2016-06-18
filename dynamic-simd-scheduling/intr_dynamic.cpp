/**********************************************************
 *
 * Copyright (c) 2016 
 * Olaf Krzikalla, olaf.krzikalla@tu-dresden.de
 * Florian Wende, wende@zib.de 
 * Markus Höhnerbach, hoehnerbach@aices.rwth-aachen.de
 *
 * The intr_dynamic approach:
 * Vectorization approach: intrinsics
 * Scheduling approach: dynamic blöcked/smoothed
 *
 * compile with -DSMOOTH for smooth scheduling
 *
 **********************************************************/

#include "main.h"
#include <immintrin.h> 

//-----------------------------------------------------------------------------
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
#define INIT_BLOCK(i)                                                          \
  __m512 re_c_##i =                                                            \
      _mm512_add_ps(vx1, _mm512_mul_ps(vdx, _mm512_my_cvtepi32_ps(i_##i)));    \
  __m512 re_z_##i = _mm512_setzero_ps();                                       \
  __m512 im_z_##i = _mm512_setzero_ps();                                       \
  __mmask16 loop_##i = _mm512_cmplt_epi32_mask(i_##i, vwidth);                 \
  __m512i k_##i = _mm512_setzero_epi32();                                      \
  __mmask16 end_bound_##i = 0xFF;                                              \
  __mmask16 end_it_##i = 0xFF;                                                 \
  __mmask16 end_##i = _mm512_kand(end_bound_##i, end_it_##i);                  \
  __m512 re_z2_##i, im_z2_##i, norm2_##i;

//-----------------------------------------------------------------------------
#define COMPUTE_UNROLL_BLOCK(i)                                                \
  k_##i = _mm512_mask_add_epi32(k_##i, end_##i, k_##i, v1);                    \
  im_z_##i = _mm512_mask_add_ps(                                               \
      im_z_##i, end_##i,                                                       \
      _mm512_mul_ps(_mm512_add_ps(re_z_##i, re_z_##i), im_z_##i), im_c);       \
  re_z_##i = _mm512_mask_add_ps(re_z_##i, end_##i, re_z2_##i,                  \
                                _mm512_sub_ps(re_c_##i, im_z2_##i));           \
  re_z2_##i = _mm512_mul_ps(re_z_##i, re_z_##i);                               \
  im_z2_##i = _mm512_mul_ps(im_z_##i, im_z_##i);                               \
  norm2_##i = _mm512_add_ps(re_z2_##i, im_z2_##i);                             \
  end_bound_##i = _mm512_cmple_ps_mask(norm2_##i, boundary);                   \
  end_it_##i = _mm512_cmplt_epi32_mask(k_##i, vmaxIters);                      \
  end_##i = _mm512_kand(end_bound_##i, end_it_##i);

//-----------------------------------------------------------------------------
#ifndef REGISTER_BLOCK
#define end_1 end_0
#else
#define end_1                                                                  \
  _mm512_knot(_mm512_kor(_mm512_kor(_mm512_knot(end_1), _mm512_knot(end_2)),   \
                         _mm512_knot(end_3)))
#endif

//-----------------------------------------------------------------------------
#ifndef REGISTER_BLOCK
#define loop_1 loop_0
#else
#define loop_1 _mm512_kor(_mm512_kor(loop_1, loop_2), loop_3)
#endif

//-----------------------------------------------------------------------------
#ifdef SMOOTH
#ifdef __MIC__
#define COUNTER_BLOCK(i)                                                       \
  i_##i = _mm512_mask_loadunpacklo_epi32(i_##i, end_##i, tmp);                 \
  _mm512_store_epi32(tmp,                                                      \
                     _mm512_add_epi32(_mm512_set1_epi32(_popcnt32(end_##i)),   \
                                      _mm512_load_epi32(tmp)));
#else
#define COUNTER_BLOCK(i)                                                       \
  i_##i = _mm512_mask_expand_epi32(i_##i, end_##i, next);                      \
  next = _mm512_add_epi32(next, _mm512_set1_epi32(_popcnt32(end_##i)));
#endif
#else
#ifdef REGISTER_BLOCK
#define COUNTER_BLOCK(i)                                                       \
  i_##i = _mm512_mask_add_epi32(i_##i, end_##i, i_##i, v64);
#else
#define COUNTER_BLOCK(i)                                                       \
  i_##i = _mm512_mask_add_epi32(i_##i, end_##i, i_##i, v16);
#endif
#endif

//-----------------------------------------------------------------------------
#define UPDATE_BLOCK(i)                                                        \
  if (__builtin_expect(                                                        \
          _mm512_kortestz(_mm512_knot(end_##i), _mm512_knot(end_##i)) == 0,    \
          0)) {                                                                \
    end_##i = _mm512_knot(end_##i);                                            \
    _mm512_mask_i32scatter_epi32(image_j, _mm512_kand(loop_##i, end_##i),      \
                                 i_##i, k_##i, 4);                             \
    COUNTER_BLOCK(i)                                                           \
    k_##i = _mm512_mask_blend_epi32(end_##i, k_##i, _mm512_setzero_epi32());   \
    re_z_##i = _mm512_mask_blend_ps(end_##i, re_z_##i, _mm512_setzero_ps());   \
    im_z_##i = _mm512_mask_blend_ps(end_##i, im_z_##i, _mm512_setzero_ps());   \
    re_c_##i =                                                                 \
        _mm512_add_ps(vx1, _mm512_mul_ps(vdx, _mm512_my_cvtepi32_ps(i_##i)));  \
    re_z2_##i = _mm512_mul_ps(re_z_##i, re_z_##i);                             \
    im_z2_##i = _mm512_mul_ps(im_z_##i, im_z_##i);                             \
    norm2_##i = _mm512_add_ps(re_z2_##i, im_z2_##i);                           \
    loop_##i = _mm512_cmplt_epi32_mask(i_##i, vwidth);                         \
    end_bound_##i = _mm512_cmple_ps_mask(norm2_##i, boundary);                 \
    end_it_##i = _mm512_cmplt_epi32_mask(k_##i, vmaxIters);                    \
    end_##i = _mm512_kand(end_bound_##i, end_it_##i);                          \
  }

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
  __m512i v32 = _mm512_set1_epi32(32);
  __m512i v64 = _mm512_set1_epi32(64);
  int *tmp = (int *)_mm_malloc(16 * sizeof(*tmp), 64);
  for (int j = 0; j < height; ++j) {
    __m512 im_c = _mm512_set1_ps(y1 + dy * j);
    __m512i i_0 = asc;
#ifdef REGISTER_BLOCK
    __m512i i_1 = _mm512_add_epi32(v16, i_0);
    __m512i i_2 = _mm512_add_epi32(v16, i_1);
    __m512i i_3 = _mm512_add_epi32(v16, i_2);
#endif

    INIT_BLOCK(0)
#ifdef REGISTER_BLOCK
    INIT_BLOCK(1)
    INIT_BLOCK(2)
    INIT_BLOCK(3)
#endif
    int *image_j = image + j * width;
    int ii = 0;

#ifdef REGISTER_BLOCK
#ifdef __MIC__
    _mm512_store_epi32(tmp, _mm512_add_epi32(v16, i_3));
#else
    __m512i next = _mm512_add_epi32(v16, i_3);
#endif
#else
#ifdef __MIC__
    _mm512_store_epi32(tmp, _mm512_add_epi32(v16, i_0));
#else
    __m512i next = _mm512_add_epi32(v16, i_0);
#endif
#endif
    if (_mm512_kortestz(loop_0, loop_1))
      continue;
    for (;;) {
#ifndef REGISTER_BLOCK
      COMPUTE_UNROLL_BLOCK(0)
      COMPUTE_UNROLL_BLOCK(0)
      COMPUTE_UNROLL_BLOCK(0)
#else
      COMPUTE_UNROLL_BLOCK(0)
      COMPUTE_UNROLL_BLOCK(1)
      COMPUTE_UNROLL_BLOCK(2)
      COMPUTE_UNROLL_BLOCK(3)
      COMPUTE_UNROLL_BLOCK(0)
      COMPUTE_UNROLL_BLOCK(1)
      COMPUTE_UNROLL_BLOCK(2)
      COMPUTE_UNROLL_BLOCK(3)
      COMPUTE_UNROLL_BLOCK(0)
      COMPUTE_UNROLL_BLOCK(1)
      COMPUTE_UNROLL_BLOCK(2)
      COMPUTE_UNROLL_BLOCK(3)
      COMPUTE_UNROLL_BLOCK(0)
      COMPUTE_UNROLL_BLOCK(1)
      COMPUTE_UNROLL_BLOCK(2)
      COMPUTE_UNROLL_BLOCK(3)
#endif
      if (__builtin_expect(
              _mm512_kortestz(_mm512_knot(end_0), _mm512_knot(end_1)) == 0,
              0)) {
        UPDATE_BLOCK(0)
#ifdef REGISTER_BLOCK
        UPDATE_BLOCK(1)
        UPDATE_BLOCK(2)
        UPDATE_BLOCK(3)
#endif
        if (_mm512_kortestz(loop_0, loop_1))
          break;
      }
    }
  }
}

