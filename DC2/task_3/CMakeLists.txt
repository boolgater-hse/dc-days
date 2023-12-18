cmake_minimum_required(VERSION 3.22)
project(task_3 CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(task_3 main.cu)

set_target_properties(task_3 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
