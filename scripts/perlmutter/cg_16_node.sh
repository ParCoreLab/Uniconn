#!/bin/bash -l

#SBATCH -N 16
#SBATCH -n 64
#SBATCH -C gpu
#SBATCH -G 64
#SBATCH -q regular
#SBATCH -J cg_16
#SBATCH -A {account_name}
#SBATCH -t 1:00:00
#SBATCH --output=results/perlmutter/%x_%j.csv
#SBATCH --error=results/perlmutter/errors/%x-%j-stderr.txt

Uniconn_Dir=
Dataset_Dir=$Uniconn_Dir/datasets/
source $Uniconn_Dir/scripts/perlmutter/env.sh
cd $Uniconn_Dir/build/examples/CG-cuda

iter_max=10

echo "benchmark, library, matrix_name, GPU_count, ms"
for mtx in $Dataset_Dir/*.mtx; do
    iter=0
    while [ $iter -lt $iter_max ]; do
        srun -n 64 -c 32 -G 64 --cpu-bind=cores standard_mpi $mtx
        srun -n 64 -c 32 -G 64 --cpu-bind=cores standard_uniconn_mpi $mtx
        
        srun -n 64 -c 32 -G 64 --cpu-bind=cores standard_gpuccl $mtx
        srun -n 64 -c 32 -G 64 --cpu-bind=cores standard_uniconn_gpuccl $mtx
        
        srun -n 64 -c 32 -G 64 --cpu-bind=cores standard_nvshmem_host $mtx
        srun -n 64 -c 32 -G 64 --cpu-bind=cores standard_uniconn_nvshmem_host $mtx
        
        srun -n 64 -c 32 -G 64 --cpu-bind=cores standard_nvshmem_device $mtx
        srun -n 64 -c 32 -G 64 --cpu-bind=cores standard_uniconn_nvshmem_device $mtx
        iter=$((iter + 1))
    done
done
