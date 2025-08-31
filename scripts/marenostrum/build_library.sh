#!/bin/bash

Uniconn_Dir=
source $Uniconn_Dir/scripts/marenostrum/env.sh

cd $Uniconn_Dir
rm -rf build
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DUNICONN_ENABLE_CUDA=ON \
    -DUNICONN_ENABLE_GPUCCL=ON -DUNICONN_ENABLE_GPUSHMEM=ON -DUNICONN_ENABLE_EXAMPLES=ON \
    -DCMAKE_INSTALL_PREFIX=install -DCMAKE_CUDA_ARCHITECTURES="90"
cd build
make -j install
