# Add executable
add_executable(sgemm_smem_transpose src/sgemm_smem_transpose.cu)
# Set CUDA architectures for RTX 4090
set_property(TARGET sgemm_smem_transpose PROPERTY CUDA_ARCHITECTURES 89)

# Set CUDA include paths
target_include_directories(sgemm_smem_transpose PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Link CUDA libraries
target_link_libraries(sgemm_smem_transpose PRIVATE CUDA::cudart) 

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(sgemm_smem_transpose PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>) 
endif() 
