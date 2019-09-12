cmake_minimum_required(VERSION 3.0)

project(finalfusion_cxx)

enable_testing()

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(finalfusion-ffi)
add_subdirectory(src)
add_subdirectory(tests)
