/**********************************************************
 *
 * Copyright (c) 2016 
 * Olaf Krzikalla, olaf.krzikalla@tu-dresden.de
 * Florian Wende, wende@zib.de 
 * Markus Höhnerbach, hoehnerbach@aices.rwth-aachen.de
 *
 * The aav_dynamic_blocked approach:
 * Vectorization approach: array_as_value
 * Scheduling approach: dynamic blocked
 *
 * this implementation restricts dynamic vector lane scheduling 
 * to single lines thus enabling a simple omp parallelization 
 * compile with -DWITH_OMP to turn on omp parallelization
 * 
 **********************************************************/

#include "array_as_value.h"
#include "simple_complex.h"
#include "main.h"

//-----------------------------------------------------------------------------
typedef array_notation_wrapper::array_as_value<float, VL> float_array;
typedef array_notation_wrapper::array_as_value<int, VL> int_array;

//-----------------------------------------------------------------------------
void Mandelbrot(float x1, float y1, float x2, float y2, int width, int height, int maxIters, int * image)
{
  float dx = (x2-x1)/width, dy = (y2-y1)/height;
  float_array real_stride(float_array::linear(dx));

#ifdef WITH_OMP
#pragma omp parallel for schedule(dynamic, 2)
#endif
  for (int j = 0; j < height; ++j, image += width)
  {
    float_array imag_c(y1+dy*j);
    int_array i (int_array::linear(1));
    complex<float_array> c (x1 + real_stride, imag_c), z(float_array(0),float_array(0));
    int_array count_array (0);
    while (i.reduce_min() < width)
    {
      float_array::mask_type mask = norm(z) < float_array(4.0f);
      if ((mask.ALLD == 0 || count_array.ALLD == maxIters) &&
          i.ALLD < width)
      {
        image[i.ALLD] = count_array.ALLD;
        i.ALLD += int_array::size;     
        count_array.ALLD = 0;
        z._Val[0].ALLD = 0;
        z._Val[1].ALLD = 0;
        c._Val[0].ALLD = x1 + i.ALLD * dx;
      }
      count_array += 1;
      z = z*z+c;
    }
  }
}

