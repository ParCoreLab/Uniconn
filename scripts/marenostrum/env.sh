#!/bin/bash
# ml cmake/3.30.2
# ml PrgEnv-nvidia
# ml gpu
# ml cudatoolkit/12.4
# ml craype-accel-nvidia80
# ml nccl/2.24.3
# #ml nvshmem/3.2.5-1

module load cmake/3.30.5
module use /apps/ACC/NVIDIA-HPC-SDK/25.1/modulefiles
module load nvhpc-hpcx-cuda12
export CUDA_HOME=${NVHPC_ROOT}/cuda/12.6
# export LD_PRELOAD=${HOME}/pkgs/glibc-2.29/lib/libm.so.6
# module use /gpfs/projects/ehpc114/jtrotter/modulefiles
export CMAKE_PREFIX_PATH=/apps/ACC/NVIDIA-HPC-SDK/25.1/Linux_x86_64/25.1/comm_libs/nvshmem/lib/cmake/nvshmem:$CMAKE_PREFIX_PATH
# environment
export LC_ALL=C
export OMP_DISPLAY_ENV=false
export OMP_NUM_THREADS=20
# export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export SLURM_CPU_BIND=none
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_IB_ENABLE_IBGDA=0
export NVSHMEM_DISABLE_NCCL=true
export NVSHMEM_DISABLE_CUDA_VMM=1
