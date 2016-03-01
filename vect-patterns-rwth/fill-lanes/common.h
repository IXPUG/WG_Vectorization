#include<math.h>
#if 1
typedef float real;
#define expr expf
#define cosr cosf
#define sinr sinf
#define sqrtr sqrtf
#define VL 16
#define MODE_SINGLE
#else
typedef double real;
#define expr exp
#define cosr cos
#define sinr sin
#define sqrtr sqrt
#define VL 8
#define MODE_DOUBLE
#endif
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
