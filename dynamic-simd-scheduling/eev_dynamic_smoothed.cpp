/**********************************************************
 *
 * Copyright (c) 2016 
 * Olaf Krzikalla, olaf.krzikalla@tu-dresden.de
 * Florian Wende, wende@zib.de 
 * Markus Höhnerbach, hoehnerbach@aices.rwth-aachen.de
 *
 * The eev_dynamic_smoothed approach:
 * Vectorization approach: enhanced explicit vectorization
 * Scheduling approach: dynamic smoothed
 *
 * this implementation restricts dynamic vector lane scheduling 
 * to single lines thus enabling a simple omp parallelization 
 * compile with -DWITH_OMP to turn on omp parallelization
 *
 **********************************************************/

#include "main.h"

//-----------------------------------------------------------------------------
#ifndef CMAX
# define CMAX 1
#endif

//-----------------------------------------------------------------------------
#define BOOL int
#define TRUE 0x1
#define FALSE 0x0

//-----------------------------------------------------------------------------
void Mandelbrot(float x1, float y1, float x2, float y2, int width, int height, int maxIter, int * image)
{
  const float dx = (x2 - x1) / width;
  const float dy = (y2 - y1) / height;
  
 
#ifdef WITH_OMP
#pragma omp parallel for schedule(dynamic, 2)
#endif
  for (int j = 0; j < height; j++)
  {
    float c_re[VL] __attribute__((aligned(ALIGNMENT)));
    float c_im[VL] __attribute__((aligned(ALIGNMENT)));
    float z_re[VL] __attribute__((aligned(ALIGNMENT)));
    float z_im[VL] __attribute__((aligned(ALIGNMENT)));
  
    int i[VL] __attribute__((aligned(ALIGNMENT)));
    int i_new[VL] __attribute__((aligned(ALIGNMENT))); 
    int count[VL] __attribute__((aligned(ALIGNMENT)));
    BOOL lane_alife[VL] __attribute__((aligned(ALIGNMENT)));
    BOOL lane_acquire_work[VL] __attribute__((aligned(ALIGNMENT)));

    int current = VL; 

  #pragma omp simd simdlen(VL)
  	for (int ii = 0; ii < VL; ii++) {
  	  c_re[ii] = x1 + dx * ii;
  	  c_im[ii] = y1 + dy * j;
  	  z_re[ii] = 0.0F;
  	  z_im[ii] = 0.0F;
      lane_alife[ii] = (ii < width ? TRUE : FALSE);
      lane_acquire_work[ii] = FALSE;
      count[ii] = 0;
      i[ii] = ii; 
  	}

    BOOL lane_acquire_work_any;
    BOOL lane_alife_any = TRUE;

    while (lane_alife_any) {

      lane_acquire_work_any = FALSE;
#pragma omp simd simdlen(VL)
      for (int ii = 0; ii < VL; ii++) {
        int temp_count = count[ii];
        float temp_z_im = z_im[ii];
        float temp_z_re = z_re[ii];
        float temp_z_im2 = temp_z_im * temp_z_im;
        float temp_z_re2 = temp_z_re * temp_z_re;
        const float temp_c_re = c_re[ii];
        const float temp_c_im = c_im[ii];
#pragma novector
        for (int c = 0; c < (CMAX - 1); c++) {
          if (temp_count < maxIter && (temp_z_im2 + temp_z_re2) < 4.0F) {
            temp_count++;
            temp_z_im = 2.0F * temp_z_im * temp_z_re + temp_c_im;
            temp_z_re = temp_z_re2 - temp_z_im2 + temp_c_re;
            temp_z_im2 = temp_z_im * temp_z_im;
            temp_z_re2 = temp_z_re * temp_z_re;
          }
        }
        if (temp_count < maxIter && (temp_z_im2 + temp_z_re2) < 4.0F) {
          temp_count++;
          temp_z_im = 2.0F * temp_z_im * temp_z_re + temp_c_im;
          temp_z_re = temp_z_re2 - temp_z_im2 + temp_c_re;
          temp_z_im2 = temp_z_im * temp_z_im;
          temp_z_re2 = temp_z_re * temp_z_re;
        } else {
          if (lane_alife[ii]) {
            lane_acquire_work[ii] = TRUE;
            lane_acquire_work_any = TRUE;
          }
        }
        z_im[ii] = temp_z_im;
        z_re[ii] = temp_z_re;
        count[ii] = temp_count;
      }

      if (lane_acquire_work_any) {
        for (int ii = 0; ii < VL; ii++)
          if (lane_acquire_work[ii])
            i_new[ii] = current++; 
            
#pragma omp simd simdlen(VL)
        for (int ii = 0; ii < VL; ii++) {
          int temp_i = i[ii];
          if (lane_acquire_work[ii]) {
            image[j * width + temp_i] = count[ii];
            count[ii] = 0;
            z_re[ii] = 0.0F;
            z_im[ii] = 0.0F;
            temp_i = i_new[ii];
            c_re[ii] = x1 + dx * temp_i;
            if (temp_i >= width)
              lane_alife[ii] = FALSE;
            i[ii] = temp_i;
          }
          lane_acquire_work[ii] = FALSE;
        }
      }

      lane_alife_any = FALSE;
#pragma omp simd simdlen(VL)
      for (int ii = 0; ii < VL; ii++)
        if (lane_alife[ii])
          lane_alife_any = TRUE; 
          
    }            
  }
}

