Dynamic SIMD Vector Lane Scheduling: Mandelbrot algorithms
==========================================================

Authors:
Olaf Krzikalla (olaf.krzikalla@tu-dresden.de)
Florian Wende (wende@zib.de)
Markus Höhnerbach (hoehnerbach@aices.rwth-aachen.de)


The source files complement the paper "Dynamic SIMD Vector Lane Scheduling" 
by the aforementioned authors. You can compile each file to a separate executable:

icc -std=c++11 -O3 -mmic -fopenmp [name].cpp [-o target_name]


