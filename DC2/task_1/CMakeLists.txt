cmake_minimum_required(VERSION 3.22)
project(task_1 C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_COMPILER mpicc)

add_executable(task_1 main.c)
target_link_libraries(task_1 m)
