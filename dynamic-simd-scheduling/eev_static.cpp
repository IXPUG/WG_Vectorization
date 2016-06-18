/**********************************************************
 *
 * Copyright (c) 2016 
 * Olaf Krzikalla, olaf.krzikalla@tu-dresden.de
 * Florian Wende, wende@zib.de 
 * Markus Höhnerbach, hoehnerbach@aices.rwth-aachen.de
 *
 * The eev_static approach:
 * Vectorization approach: enhanced explicit vectorization
 * Scheduling approach: static
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
    int count[VL] __attribute__((aligned(ALIGNMENT)));
    BOOL m_1[VL];
    
    for (int i = 0; i < width; i+=VL) 
    {
    #pragma omp simd simdlen(VL)
    	for (int ii = 0; ii < VL; ii++) {
    	  c_re[ii] = x1 + dx * (i + ii);
    	  c_im[ii] = y1 + dy * j;
    	  z_re[ii] = 0.0F;
    	  z_im[ii] = 0.0F;
    	  count[ii] = 0;
    	  m_1[ii] = (i + ii) < width ? TRUE : FALSE;
    	}

    	BOOL true_for_any = TRUE;
    	while(true_for_any) {
    	  true_for_any = FALSE;
    #pragma omp simd simdlen(VL)
    	  for (int ii = 0; ii < VL; ii++) {
    	    float temp_z_re = z_re[ii];
    	    float temp_z_im = z_im[ii];
    	    float temp_z_re2 = temp_z_re * temp_z_re;
    	    float temp_z_im2 = temp_z_im * temp_z_im;
    	    int temp_count = count[ii];
    	    BOOL temp_m_1 = m_1[ii];
    	    for (int c = 0; c < (CMAX - 1); c++) {
    	      if (temp_m_1 && (temp_count < maxIter) && (temp_z_re2 + temp_z_im2) < 4.0F) {
          		temp_count++;
          		temp_z_im = 2.0F * temp_z_re * temp_z_im + c_im[ii];
          		temp_z_re = temp_z_re2 - temp_z_im2 + c_re[ii];
          		temp_z_re2 = temp_z_re * temp_z_re;
          		temp_z_im2 = temp_z_im * temp_z_im;
    	      }
    	    }
    	    if (temp_m_1 && (temp_count < maxIter) && (temp_z_re2 + temp_z_im2) < 4.0F) {
    	      temp_count++;
    	      temp_z_im = 2.0F * temp_z_re * temp_z_im + c_im[ii];
    	      temp_z_re = temp_z_re2 - temp_z_im2 + c_re[ii];
    	      true_for_any = TRUE;
    	    }
    	    z_re[ii] = temp_z_re;
    	    z_im[ii] = temp_z_im;
    	    count[ii] = temp_count;
    	  }
    	}
  	
    #pragma omp simd simdlen(VL)
    	for (int ii = 0; ii < VL; ii++) {
    	  if (m_1[ii])
    	    image[j * width + i + ii] = count[ii];
    	}
    }
  }
}

