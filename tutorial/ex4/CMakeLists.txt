cmake_minimum_required(VERSION 3.20)

# 判断如果使用的是 MSVC 编译器，则添加 /utf-8 参数
if (MSVC)
    add_compile_options(/utf-8)
endif()
#set(CMAKE_CUDA_ARCHITECTURES 89)

project(CudaGPU2 CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(CudaGPU2
        main.cu
        src/Tools.cuh
        src/Reduce.cuh
)

set_target_properties(CudaGPU2 PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
