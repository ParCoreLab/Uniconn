#!/bin/bash -l
module load LUMI/24.03
module load partition/G
module load EasyBuild-user
module load aws-ofi-rccl/17d41cb-cpeAMD-24.03
module load PrgEnv-amd
module load rocm/6.0.3
module load craype-accel-amd-gfx90a
module load buildtools/24.03

export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:$LD_LIBRARY_PATH
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_IPC_THRESHOLD=1
# export MPICH_MAX_THREAD_SAFETY=multiple
# export MPICH_ASYNC_PROGRESS=0
export MPICH_OFI_NIC_POLICY=GPU
# export MPICH_SMP_SINGLE_COPY_MODE=COPY

export FI_CXI_ATS=0
export FI_CXI_DISABLE_HOST_REGISTER=1
# export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_OPTIMIZED_MRS=false
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216

export HSA_ENABLE_SDMA=0 # enabling SDMA speeds up asynchronous device-to-host transfers
export HSA_ENABLE_PEER_SDMA=0
export HSA_FORCE_FINE_GRAIN_PCIE=1

export NCCL_MIN_CTAS=32
export NCCL_P2P_LEVEL=4
export NCCL_NET_GDR_LEVEL=3
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_IGNORE_CPU_AFFINITY=1

 export CPU_BIND="mask_cpu:fe000000000000,fe00000000000000,fe0000,fe000000,fe,fe00,fe00000000,fe0000000000"
#export CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"