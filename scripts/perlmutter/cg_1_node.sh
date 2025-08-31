#!/bin/bash -l

#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q regular
#SBATCH -J cg_1
#SBATCH -A {account_name}
#SBATCH -t 4:00:00
#SBATCH --output=results/perlmutter/%x_%j.csv
#SBATCH --error=results/perlmutter/errors/%x-%j-stderr.txt

Uniconn_Dir=
Dataset_Dir=$Uniconn_Dir/datasets/
source $Uniconn_Dir/scripts/perlmutter/env.sh
cd $Uniconn_Dir/build/examples/CG-cuda
iter_max=10
max_gpu=4
echo "benchmark, library, matrix_name, GPU_count, ms"

for mtx in $Dataset_Dir/*.mtx; do
    iter=0
    while [ $iter -lt $iter_max ]; do
        gpu_num=1
        while [ $gpu_num -le $max_gpu ]; do
            srun -n $gpu_num -c 32 -G $gpu_num --cpu-bind=cores standard_mpi $mtx
            srun -n $gpu_num -c 32 -G $gpu_num --cpu-bind=cores standard_uniconn_mpi $mtx

            srun -n $gpu_num -c 32 -G $gpu_num --cpu-bind=cores standard_gpuccl $mtx
            srun -n $gpu_num -c 32 -G $gpu_num --cpu-bind=cores standard_uniconn_gpuccl $mtx

            srun -n $gpu_num -c 32 -G $gpu_num --cpu-bind=cores standard_nvshmem_host $mtx
            srun -n $gpu_num -c 32 -G $gpu_num --cpu-bind=cores standard_uniconn_nvshmem_host $mtx

            srun -n $gpu_num -c 32 -G $gpu_num --cpu-bind=cores standard_nvshmem_device $mtx
            srun -n $gpu_num -c 32 -G $gpu_num --cpu-bind=cores standard_uniconn_nvshmem_device $mtx
            gpu_num=$((gpu_num * 2))
        done
        iter=$((iter + 1))
    done
done
