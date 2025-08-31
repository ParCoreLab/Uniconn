#!/bin/bash
ml cmake/3.30.2
ml PrgEnv-nvidia
ml gpu
ml cudatoolkit/12.4
ml craype-accel-nvidia80
ml nccl/2.24.3
#ml nvshmem/3.2.5-1

# set all required environment variables
export CMAKE_PREFIX_PATH=/global/common/software/nersc9/nvshmem/3.2.5-1-25.03/lib/cmake/nvshmem:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=/global/common/software/nersc9/nvshmem/3.2.5-1-25.03/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/global/common/software/nersc9/nccl/2.24.3/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/global/common/software/nersc9/nccl/2.24.3/plugin/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:$LD_LIBRARY_PATH
#MPI settings:
export MPICH_GPU_SUPPORT_ENABLED=1
export CRAY_ACCEL_TARGET=nvidia80
export MPICH_GPU_IPC_THRESHOLD=1
# export MPICH_MAX_THREAD_SAFETY=multiple
# export MPICH_ASYNC_PROGRESS=1
export MPICH_OFI_NIC_POLICY=GPU
# export MPICH_SMP_SINGLE_COPY_MODE=COPY
##Libfabric settings:
export FI_CXI_OPTIMIZED_MRS=false
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216
##NCCL settings:
export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_SOCKET_IFNAME=hsn
export NCCL_IGNORE_CPU_AFFINITY=1
##NVSHMEM settings:

export NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_REMOTE_TRANSPORT=libfabric
export CPU_BIND="map_cpu:1,16,32,48"