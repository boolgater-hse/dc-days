#!/usr/bin/bash

set -e

BUILD_DIR=cmake-build-debug
mkdir -p $BUILD_DIR
cd $BUILD_DIR
cmake ..
cd ..
cmake --build $BUILD_DIR --target task_1 -- -j 14
mpiexec -np $1 $BUILD_DIR/task_1
