#!/usr/bin/bash

set -e

BUILD_DIR=cmake-build-release
mkdir -p $BUILD_DIR
cd $BUILD_DIR
cmake ..
cd ..
cmake --build $BUILD_DIR --target task_2 -- -j 14
mpiexec -np $1 $BUILD_DIR/task_2
