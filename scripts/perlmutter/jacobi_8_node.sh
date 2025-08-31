#!/bin/bash -l

#SBATCH -N 8
#SBATCH -C gpu
#SBATCH -G 32
#SBATCH -q regular
#SBATCH -J jacobi_8
#SBATCH -A {account_name}
#SBATCH -t 1:00:00
#SBATCH --output=results/perlmutter/%x_%j.csv
#SBATCH --error=results/perlmutter/errors/%x-%j-stderr.txt

Uniconn_Dir=
source $Uniconn_Dir/scripts/perlmutter/env.sh
cd $Uniconn_Dir/build/examples/jacobi2D
iter_max=10

echo "benchmark, library, GPU_count, ms"
iter=0
while [ $iter -lt $iter_max ]; do
    srun -n 32 -c 32 -G 32 --cpu-bind=cores single_stream_mpi
    srun -n 32 -c 32 -G 32 --cpu-bind=cores single_stream_uniconn_mpi

    srun -n 32 -c 32 -G 32 --cpu-bind=cores single_stream_gpuccl
    srun -n 32 -c 32 -G 32 --cpu-bind=cores single_stream_uniconn_gpuccl

    srun -n 32 -c 32 -G 32 --cpu-bind=cores single_stream_nvshmem_host
    srun -n 32 -c 32 -G 32 --cpu-bind=cores single_stream_uniconn_nvshmem_host

    srun -n 32 -c 32 -G 32 --cpu-bind=cores single_stream_nvshmem_device
    srun -n 32 -c 32 -G 32 --cpu-bind=cores single_stream_uniconn_nvshmem_device
    iter=$((iter + 1))
done
