cmake_minimum_required(VERSION 3.30)

# 判断如果使用的是 MSVC 编译器，则添加 /utf-8 参数
if (MSVC)
    add_compile_options(/utf-8)
endif()


include_directories(./src/Common)

project(CudaGPU2 CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(CudaGPU2
        main.cu
)

set_target_properties(CudaGPU2 PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
