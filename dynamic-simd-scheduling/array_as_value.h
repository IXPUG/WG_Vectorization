/**********************************************************
 *
 * Copyright (c) 2015 ZIH, TU Dresden
 * Olaf Krzikalla, olaf.krzikalla@tu-dresden.de
 *
 * This file is a copy to prevent dependencies to other 
 * projects. Visit http://tu-dresden.de/zih/array-as-value 
 * for further information.
 *
 **********************************************************/

#ifndef ANW_ARRAY_AS_VALUE_H
#define ANW_ARRAY_AS_VALUE_H

#include <cstddef>
#include <memory>
#include <aligned_new>
#include <initializer_list>

namespace array_notation_wrapper
{

#define ALLD data_[:]

#define ARRAY_AS_VALUE_OPERATOR( op ) \
  friend array_as_value operator op (array_as_value x, const array_as_value& y)  { return x op##= y; } \
  friend array_as_value operator op (array_as_value x, const T& y)  { return x op##= y; } \
  friend array_as_value operator op (const T& x, const array_as_value& y)  { array_as_value result; result.ALLD = x op y.ALLD; return result; } \
  array_as_value& operator op##=(const array_as_value& x) { ALLD op##= x.ALLD; return *this; } \
  array_as_value& operator op##=(const T& x)  { ALLD op##= x; return *this; }

#define ARRAY_AS_VALUE_BOOL_OPERATOR( op ) \
  friend mask_type operator op (const array_as_value& x, const array_as_value& y)  { mask_type result; result.ALLD = x.ALLD op y.ALLD; return result; }
  
#define ARRAY_AS_VALUE_FUNCTION( FUNC ) \
  template<class T, size_t SZ, size_t ALIGN> \
  inline array_notation_wrapper::array_as_value<T, SZ, ALIGN> FUNC(const array_notation_wrapper::array_as_value<T, SZ, ALIGN>& x) { \
    array_notation_wrapper::array_as_value<T, SZ, ALIGN> result; \
    result.ALLD = FUNC(x.ALLD); \
    return result; }

#define ARRAY_AS_VALUE_REDUCER( FUNC ) \
  decltype(__sec_##FUNC(data_[:])) FUNC() const { return __sec_##FUNC(data_[:]); }


//--------------------------------------------------------
template<class T, size_t SZ, size_t ALIGN = 0> 
struct array_as_value 
{
  typedef T value_type;
  enum { size = SZ };
  typedef array_as_value<int, SZ> mask_type;
  typedef array_as_value<int, SZ> index_type;

  alignas(ALIGN) T data_[SZ];

  array_as_value() { }

  // construct array_as_value{ init, init ..., init }
  explicit array_as_value(const T& init) 
  {
    ALLD = init;
  }

  array_as_value(std::initializer_list<T> init)
  {
    ALLD = init.begin()[0:SZ];
  } 

  array_as_value(const array_as_value& x) 
  {
    ALLD = x.ALLD;
  }

  array_as_value& operator=(const array_as_value& x) 
  {
    ALLD = x.ALLD;
    return *this;
  }

  const T& operator[](size_t i) const { return data_[i]; }
  T& operator[](size_t i) { return data_[i]; }

  ARRAY_AS_VALUE_OPERATOR( * )
  ARRAY_AS_VALUE_OPERATOR( / )
  ARRAY_AS_VALUE_OPERATOR( + )
  ARRAY_AS_VALUE_OPERATOR( - )

  ARRAY_AS_VALUE_BOOL_OPERATOR( == )
  ARRAY_AS_VALUE_BOOL_OPERATOR( != )
  ARRAY_AS_VALUE_BOOL_OPERATOR( < )
  ARRAY_AS_VALUE_BOOL_OPERATOR( > )
  ARRAY_AS_VALUE_BOOL_OPERATOR( >= )
  ARRAY_AS_VALUE_BOOL_OPERATOR( <= )
  
  // be careful: the arguments x and y are fully evaluated 
  friend array_as_value ternary(const typename array_as_value::mask_type& m, array_as_value x, const array_as_value& y)
  { 
    x.ALLD = m.ALLD ? x.ALLD : y.ALLD;
    return x;
  }

  array_as_value operator-() const
  { 
    array_as_value result;
    result.ALLD = -ALLD; 
    return result;
  }
  
  ARRAY_AS_VALUE_REDUCER(reduce_add)
  ARRAY_AS_VALUE_REDUCER(reduce_mul)
  ARRAY_AS_VALUE_REDUCER(reduce_min)
  ARRAY_AS_VALUE_REDUCER(reduce_max)
  ARRAY_AS_VALUE_REDUCER(reduce_min_ind)
  ARRAY_AS_VALUE_REDUCER(reduce_max_ind)
  ARRAY_AS_VALUE_REDUCER(reduce_all_zero)
  ARRAY_AS_VALUE_REDUCER(reduce_all_nonzero)
  ARRAY_AS_VALUE_REDUCER(reduce_any_zero)
  ARRAY_AS_VALUE_REDUCER(reduce_any_nonzero)

  // returns array_as_value{ ar[index[0]], ar[index[1]], ..., ar[index[SIZE-1]] }
  static array_as_value gather(const value_type* ar, const index_type& index)
  {
    array_as_value result;
    result.ALLD = ar[index.ALLD];
    return result;
  }

  // returns array_as_value{ 0, increment, increment*2, ..., increment*(SIZE-1) }
  static array_as_value linear(value_type increment)
  {
    array_as_value result;
    result.data_[:] = increment * __sec_implicit_index(0);
    return result;
  }

  // returns a mask with the first len bits set (useful e.g. for residual loops)
  static mask_type residual_mask(size_t len)
  {
    mask_type result;
    result.data_[0:len] = 1;
    result.data_[len:mask_type::size-len] = 0;
    return result;
  }
};
 
//--------------------------------------------------------- 
template<size_t SZ, size_t ALIGN> 
inline array_as_value<float, SZ, ALIGN> sqrt(const array_as_value<float, SZ, ALIGN>& x) 
{ 
  array_as_value result; 
  result.ALLD = sqrtf(x.ALLD); 
  return result; 
}
  
//--------------------------------------------------------- 
template<class T>
class aligned_allocator 
{
 public:
    typedef size_t     size_type;
    typedef ptrdiff_t  difference_type;
    typedef T*         pointer;
    typedef const T*   const_pointer;
    typedef T&         reference;
    typedef const T&   const_reference;
    typedef T          value_type;

    template<typename Tp1>
    struct rebind
    { typedef aligned_allocator<Tp1> other; };

    aligned_allocator() throw() { }

    template<typename Tp1>
    aligned_allocator(const aligned_allocator<Tp1>&) throw() { }

    ~aligned_allocator() throw() { }

    pointer address(reference __x) const { return std::__addressof(__x); }

    const_pointer address(const_reference __x) const { return std::__addressof(__x); }

    // NB: __n is permitted to be 0.  The C++ standard says nothing
    // about what the return value is when __n == 0.
    pointer allocate(size_type __n, const void* = 0)
    { 
    	if (__n > this->max_size())
	      std::__throw_bad_alloc();

	    return static_cast<value_type*>(::operator new(__n * sizeof(value_type), alignof(T)));
    }

      // __p is not permitted to be a null pointer.
      void deallocate(pointer __p, size_type)
      { ::operator delete(__p); }

      size_type
      max_size() const throw() 
      { return size_t(-1) / sizeof(value_type); }

      void 
      construct(pointer __p, const value_type& __val) 
      { ::new((void *)__p) value_type(__val); }

      void 
      destroy(pointer __p) { __p->~value_type(); }
};

} // namespace array_notation_wrapper 

//--------------------------------------------------------- 
// defined outside of namespace array_notation_wrapper, so that
// the scalar functions can be found:
ARRAY_AS_VALUE_FUNCTION(sqrt)
ARRAY_AS_VALUE_FUNCTION(fabs)
ARRAY_AS_VALUE_FUNCTION(exp)
ARRAY_AS_VALUE_FUNCTION(sin)
ARRAY_AS_VALUE_FUNCTION(cos)
ARRAY_AS_VALUE_FUNCTION(tan)

//--------------------------------------------------------- 
template<class T, size_t SZ, size_t ALIGN>
class std::allocator<array_notation_wrapper::array_as_value<T, SZ, ALIGN> > :
  public array_notation_wrapper::aligned_allocator<array_notation_wrapper::array_as_value<T, SZ, ALIGN> >
{};

#endif //ANW_ARRAY_AS_VALUE_H

/** end of file *******************************************/
