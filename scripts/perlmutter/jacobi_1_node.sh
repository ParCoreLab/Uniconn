#!/bin/bash -l

#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q regular
#SBATCH -J jacobi_1
#SBATCH -A {account_name}
#SBATCH -t 4:00:00
#SBATCH --output=results/perlmutter/%x_%j.csv
#SBATCH --error=results/perlmutter/errors/%x-%j-stderr.txt

Uniconn_Dir=
source $Uniconn_Dir/scripts/perlmutter/env.sh
cd $Uniconn_Dir/build/examples/jacobi2D
max_gpu=4
iter_max=10

echo "benchmark, library, GPU_count, ms"
iter=0
while [ $iter -lt $iter_max ]; do
    gpu_num=1
    while [ $gpu_num -le $max_gpu ]; do
        srun -n $gpu_num -c 32 -G $gpu_num --cpu-bind=cores single_stream_mpi
        srun -n $gpu_num -c 32 -G $gpu_num --cpu-bind=cores single_stream_uniconn_mpi

        srun -n $gpu_num -c 32 -G $gpu_num --cpu-bind=cores single_stream_gpuccl
        srun -n $gpu_num -c 32 -G $gpu_num --cpu-bind=cores single_stream_uniconn_gpuccl

        srun -n $gpu_num -c 32 -G $gpu_num --cpu-bind=cores single_stream_nvshmem_host
        srun -n $gpu_num -c 32 -G $gpu_num --cpu-bind=cores single_stream_uniconn_nvshmem_host

        srun -n $gpu_num -c 32 -G $gpu_num --cpu-bind=cores single_stream_nvshmem_device
        srun -n $gpu_num -c 32 -G $gpu_num --cpu-bind=cores single_stream_uniconn_nvshmem_device
        gpu_num=$((gpu_num * 2))
    done
    iter=$((iter + 1))
done
