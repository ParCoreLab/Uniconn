#!/bin/bash -l

#SBATCH --job-name=latency_inter
#SBATCH --account={account_name}
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=0-01:00:00
#SBATCH --output=results/lumi/%x_%j.csv
#SBATCH --error=results/lumi/errors/%x-%j-stderr.txt

Uniconn_Dir=
source $Uniconn_Dir/scripts/lumi/env.sh
cd $Uniconn_Dir/build/examples/benchmarks
iter_max=10

echo "benchmark, library, buf_size, microseconds"
iter=0
while [ $iter -lt $iter_max ]; do
    srun -n 2 --ntasks-per-node=1 --gpus-per-node=1 --cpu-bind=${CPU_BIND} latency_mpi
    srun -n 2 --ntasks-per-node=1 --gpus-per-node=1 --cpu-bind=${CPU_BIND} latency_uniconn_mpi

    srun -n 2 --ntasks-per-node=1 --gpus-per-node=1 --cpu-bind=${CPU_BIND} latency_gpuccl
    srun -n 2 --ntasks-per-node=1 --gpus-per-node=1 --cpu-bind=${CPU_BIND} latency_uniconn_gpuccl
    iter=$((iter + 1))
done
