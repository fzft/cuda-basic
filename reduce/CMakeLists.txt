# Add executable
add_executable(reduce_global_memory src/reduce_global_memory.cu)
# Set CUDA architectures for RTX 4090
set_property(TARGET reduce_global_memory PROPERTY CUDA_ARCHITECTURES 89)

# Set CUDA include paths
target_include_directories(reduce_global_memory PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Link CUDA libraries
target_link_libraries(reduce_global_memory PRIVATE CUDA::cudart) 

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_global_memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>) 
endif() 
