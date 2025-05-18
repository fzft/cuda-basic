cmake_minimum_required(VERSION 3.20.0)
project(cuda_basic CUDA CXX C)

# Find CUDA package
find_package(CUDAToolkit REQUIRED)

add_subdirectory(reduce)
add_subdirectory(sgemm)

