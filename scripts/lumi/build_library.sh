#!/bin/bash

Uniconn_Dir=
source $Uniconn_Dir/scripts/lumi/env.sh

cd $Uniconn_Dir
rm -rf build
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DUNICONN_ENABLE_ROCM=ON \
 -DUNICONN_ENABLE_GPUCCL=ON -DUNICONN_ENABLE_EXAMPLES=ON -DCMAKE_INSTALL_PREFIX=install \
  -DGPU_TARGETS=gfx90a  -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=amdclang++ -DCMAKE_INSTALL_PREFIX=install \
  -DCMAKE_LINKER=CC -DAMDGPU_TARGETS=gfx90a 
cd build 
make install -j