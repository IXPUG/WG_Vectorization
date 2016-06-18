/**********************************************************
 *
 * Copyright (c) 2016 
 * Olaf Krzikalla, olaf.krzikalla@tu-dresden.de
 * Florian Wende, wende@zib.de 
 * Markus Höhnerbach, hoehnerbach@aices.rwth-aachen.de
 *
 * The aav_static approach:
 * Vectorization approach: array_as_value
 * Scheduling approach: static
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

#ifdef WITH_OMP
#pragma omp parallel for schedule(dynamic, 2)
#endif
  for (int j = 0; j < height; ++j)
  {
    float_array imag_c(y1+dy*j);
    float_array int_stride (float_array::linear(1));
    for (int i = 0; i < width; i += float_array::size, image += float_array::size, int_stride += float_array::size)
	  {
	    complex<float_array> c (x1 + int_stride * dx, imag_c), z(float_array(0),float_array(0));
	    int count = -1;
      int_array count_array (0);
	    while (++count < maxIters)
      {
        float_array::mask_type mask = norm(z) < float_array(4.0f);
        if (mask.reduce_all_zero()) break;
        count_array += mask;
        z = z*z+c;
      }
	    image[0:int_array::size] = count_array.data_[:];
	  }
  }
}

