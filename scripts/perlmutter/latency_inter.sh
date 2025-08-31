#!/bin/bash -l

#SBATCH -N 2
#SBATCH -C gpu
#SBATCH -G 2
#SBATCH -c 128
#SBATCH -n 2
#SBATCH --gpus-per-node=1
#SBATCH -q regular
#SBATCH -J latency_inter
#SBATCH -A {account_name}
#SBATCH -t 01:00:00
#SBATCH --output=results/perlmutter/%x_%j.csv
#SBATCH --error=results/perlmutter/errors/%x-%j-stderr.txt

Uniconn_Dir=
source $Uniconn_Dir/scripts/perlmutter/env.sh
cd $Uniconn_Dir/build/examples/benchmarks
iter_max=10

echo "benchmark, library, buf_size, microseconds"
iter=0
while [ $iter -lt $iter_max ]; do
srun -N 2 -n 2 --gpus-per-node=1 --ntasks-per-node=1 -c 32 --cpu-bind=cores latency_mpi 
srun -N 2 -n 2 --gpus-per-node=1 --ntasks-per-node=1 -c 32 --cpu-bind=cores latency_uniconn_mpi 

srun -N 2 -n 2 --gpus-per-node=1 --ntasks-per-node=1 -c 32 --cpu-bind=cores latency_gpuccl 
srun -N 2 -n 2 --gpus-per-node=1 --ntasks-per-node=1 -c 32 --cpu-bind=cores latency_uniconn_gpuccl 

srun -N 2 -n 2 --gpus-per-node=1 --ntasks-per-node=1 -c 32 --cpu-bind=cores latency_nvshmem_device 
srun -N 2 -n 2 --gpus-per-node=1 --ntasks-per-node=1 -c 32 --cpu-bind=cores latency_uniconn_nvshmem_device 
iter=$((iter + 1))
done