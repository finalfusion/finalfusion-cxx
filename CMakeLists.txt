cmake_minimum_required(VERSION 3.0)

project(finalfusion_cxx)

enable_testing()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(finalfusion-ffi)
add_subdirectory(finalfusion-tf)
add_subdirectory(src)
add_subdirectory(tests)
