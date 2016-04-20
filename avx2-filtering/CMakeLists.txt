cmake_minimum_required(VERSION 3.1.0)

project(avx2-filtering)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 11 CACHE STRING "C++ ISO Standard")
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo)
endif()

include(CompilerSetup)

add_compile_options(${CMAKE_CXX_FLAGS_TARGET_ISA})

add_executable(avx2mask src/avx2filter.cc)

add_executable(avx2filter src/avx2filter.cc)
target_compile_definitions(avx2filter PUBLIC -DSORT_SIMD_LANES)

add_executable(avx2filter_perm src/avx2filter.cc)
target_compile_definitions(avx2filter_perm PUBLIC -DSORT_SIMD_LANES -DPERMVAR)

add_executable(avx2filterv src/avx2filter.cc)
target_compile_definitions(avx2filterv PUBLIC -DSORT_SIMD_LANES -DVERBOSE)

add_executable(avx2filterv_perm src/avx2filter.cc)
target_compile_definitions(avx2filterv_perm PUBLIC -DSORT_SIMD_LANES -DVERBOSE -DPERMVAR)

add_executable(avx2sortbench src/bench.cc)
target_compile_definitions(avx2sortbench PUBLIC)

add_executable(avx2sortbench_perm src/bench.cc)
target_compile_definitions(avx2sortbench_perm PUBLIC -DPERMVAR)

enable_testing()

add_test(Masking    avx2mask   33554432 -1.0 1.0 0.0)
add_test(FilterSort avx2filter 33554432 -1.0 1.0 0.0)
add_test(FilterPerm avx2filter_perm 33554432 -1.0 1.0 0.0)

add_test(Example avx2filterv 64 -1.0 1.0 0.0)

