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

add_executable(avx2filterv src/avx2filter.cc)
target_compile_definitions(avx2filterv PUBLIC -DSORT_SIMD_LANES -DVERBOSE)

enable_testing()

add_test(Masking   avx2mask   33554432 -1.0 1.0 0.0)
add_test(Filtering avx2filter 33554432 -1.0 1.0 0.0)

add_test(Example avx2filterv 64 -1.0 1.0 0.0)
