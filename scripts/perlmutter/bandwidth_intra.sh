#!/bin/bash -l

#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 2
#SBATCH -n 2
#SBATCH -c 64
#SBATCH -q regular
#SBATCH -J bandwidth_intra
#SBATCH -A {account_name}
#SBATCH -t 01:00:00
#SBATCH --output=results/perlmutter/%x_%j.csv
#SBATCH --error=results/perlmutter/errors/%x-%j-stderr.txt

Uniconn_Dir=
source $Uniconn_Dir/scripts/perlmutter/env.sh
cd $Uniconn_Dir/build/examples/benchmarks

iter_max=10

echo "benchmark, library, buf_size, MB/s"
iter=0
while [ $iter -lt $iter_max ]; do
    srun -N 1 -n 2 --gpus-per-node=2 --ntasks-per-node=2 -c 32 --cpu-bind=cores bandwidth_mpi
    srun -N 1 -n 2 --gpus-per-node=2 --ntasks-per-node=2 -c 32 --cpu-bind=cores bandwidth_uniconn_mpi

    srun -N 1 -n 2 --gpus-per-node=2 --ntasks-per-node=2 -c 32 --cpu-bind=cores bandwidth_gpuccl
    srun -N 1 -n 2 --gpus-per-node=2 --ntasks-per-node=2 -c 32 --cpu-bind=cores bandwidth_uniconn_gpuccl

    srun -N 1 -n 2 --gpus-per-node=2 --ntasks-per-node=2 -c 32 --cpu-bind=cores bandwidth_nvshmem_device
    srun -N 1 -n 2 --gpus-per-node=2 --ntasks-per-node=2 -c 32 --cpu-bind=cores bandwidth_uniconn_nvshmem_device
    iter=$((iter + 1))
done
