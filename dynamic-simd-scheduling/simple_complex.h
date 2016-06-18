/**********************************************************
 *
 * Copyright (c) 2016 ZIH, TU Dresden
 * Olaf Krzikalla, olaf.krzikalla@tu-dresden.de
 *
 * This file contains a flattened version of std::complex 
 * Since _Val is directly accessible, array section 
 * operators can be applied.  
 *
 **********************************************************/

#if defined ( WIN32 )
#  include <complex>
using std::complex;
using std::norm;

#else

template<class T>
struct complex
{
  T _Val[2];
  
  complex() {}
  complex(const T& r, const T& i) { _Val[0] = r; _Val[1] = i; }
  
  const T& real() const { return _Val[0]; }
  const T& imag() const { return _Val[1]; }
  
  complex& operator+=(const complex& x) { _Val[0] += x._Val[0]; _Val[1] += x._Val[1]; return *this; } 
  complex& operator*=(const complex& x) 
  {

		T tmp = real() * x.real() - imag() * x.imag();
    _Val[1] = real() * x.imag() + imag() * x.real();
    _Val[0] = tmp;
    return *this;
  }
  
  friend complex operator+ (complex x, const complex& y)  { return x += y; } 
  friend complex operator* (complex x, const complex& y)  { return x *= y; } 
};  

template<class T>
T norm(const complex<T>& val)
{
  return val.real()*val.real() + val.imag()*val.imag(); 
}

#endif

