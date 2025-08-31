#!/bin/bash -l

#SBATCH -C gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH -q regular
#SBATCH -J latency_intra
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
srun -N 1 -n 2 --gpus-per-node=2 --ntasks-per-node=2 -c 32 --cpu-bind=cores latency_mpi 
srun -N 1 -n 2 --gpus-per-node=2 --ntasks-per-node=2 -c 32 --cpu-bind=cores latency_uniconn_mpi 

srun -N 1 -n 2 --gpus-per-node=2 --ntasks-per-node=2 -c 32 --cpu-bind=cores latency_gpuccl 
srun -N 1 -n 2 --gpus-per-node=2 --ntasks-per-node=2 -c 32 --cpu-bind=cores latency_uniconn_gpuccl 

srun -N 1 -n 2 --gpus-per-node=2 --ntasks-per-node=2 -c 32 --cpu-bind=cores latency_nvshmem_device 
srun -N 1 -n 2 --gpus-per-node=2 --ntasks-per-node=2 -c 32 --cpu-bind=cores latency_uniconn_nvshmem_device 
iter=$((iter + 1))
done