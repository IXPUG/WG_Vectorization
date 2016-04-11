#include<math.h>
#if 1
typedef float real;
#define expr expf
#define cosr cosf
#define sinr sinf
#define sqrtr sqrtf
#if defined(__MIC__) || defined(__AVX512F__)
#define VL 16
#else
#define VL 8
#endif
#define MODE_SINGLE
#else
typedef double real;
#define expr exp
#define cosr cos
#define sinr sin
#define sqrtr sqrt
#if defined(__MIC__) || defined(__AVX512F__)
#define VL 8
#else
#define VL 4
#endif
#define MODE_DOUBLE
#endif
#pragma omp declare simd
__declspec(vector)
inline void compute_f(real dij, real dik, real *fi, real *fj, real *fk) {
    dij *= 10;
    dik *= 10;
    *fi = expr(dik) * cosr(dij) * sinr(dik) * dij;
    *fj = sinr(dij) * expr(dij) * cosr(dik) * dik;
//    *fi = 1;
//    *fj = 2;
    *fk = - *fi - *fj;
}
