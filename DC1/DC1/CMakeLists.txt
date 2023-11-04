cmake_minimum_required(VERSION 3.17)
project(DC1 C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE 1)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Xclang -fopenmp")

include_directories("/opt/homebrew/opt/libomp/include")
link_directories("/opt/homebrew/opt/libomp/lib")
add_executable(DC1 main.c)
target_link_libraries(DC1 omp)
