/**********************************************************
 *
 * Copyright (c) 2016 
 * Olaf Krzikalla, olaf.krzikalla@tu-dresden.de
 * Florian Wende, wende@zib.de 
 * Markus Höhnerbach, hoehnerbach@aices.rwth-aachen.de
 *
 **********************************************************/

#if defined(__MIC__) || defined(__AVX512F__)
#  define ALIGNMENT (64)
#  define VL (16)
#else
#  ifndef VL
#    error "Define VL on platforms other than MIC"  
#  endif
#endif

#define WIDTH 1920
#define HEIGHT 1200

#include <cstdio>
#include <cstdlib>
#include "timing.h"


void Mandelbrot(float x1, float y1, float x2, float y2, int width, int height, int maxIters, int * image);


int main(int argc, char*argv[]) 
{
  float center_x = -0.74529;
  float center_y = 0.113075;
  float extrusion_x = 1.5E-4;
  float extrusion_y = 1.5E-4;
  int iter = 10000;

  if (argc > 1)
  {
    center_x = atof(argv[1]);
    center_y = atof(argv[2]);
    extrusion_x = atof(argv[3]);
    extrusion_y = atof(argv[4]);
    iter = atoi(argv[5]);
  }

  float x0 = center_x - extrusion_x;
  float x1 = center_x + extrusion_x;
  float y0 = center_y - extrusion_y;
  float y1 = center_y + extrusion_y;  
  
  int *buf = new int[WIDTH*HEIGHT];
  
  reset_and_start_timer();
  Mandelbrot(x0, y0, x1, y1, (int) WIDTH, (int) HEIGHT, iter, buf);
  double dt = get_elapsed_mcycles();
  printf("@time of Mandelbrot run:\t\t[%.3f] million cycles\n", dt);
  // check result: for default values it should be around 369816967
  int sum = __sec_reduce_add(buf[0:WIDTH*HEIGHT]);  
  printf("@Mandelbrot run:\t\t\t[%d] iterations\n", sum);
  return 0;   
}

