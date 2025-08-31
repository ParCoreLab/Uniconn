#!/bin/bash -l
#SBATCH --job-name=jacobi_8
#SBATCH --account={account_name}
#SBATCH --partition=standard-g
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=0-01:00:00
#SBATCH --output=results/lumi/%x_%j.csv
#SBATCH --error=results/lumi/errors/%x-%j-stderr.txt

Uniconn_Dir=
source $Uniconn_Dir/scripts/lumi/env.sh
cd $Uniconn_Dir/build/examples/jacobi2D

iter_max=10

echo "benchmark, library, GPU_count, ms"
iter=0
while [ $iter -lt $iter_max ]; do
    srun --cpu-bind=${CPU_BIND} -n 64 -G 64 single_stream_mpi
    srun --cpu-bind=${CPU_BIND} -n 64 -G 64 single_stream_uniconn_mpi

    srun --cpu-bind=${CPU_BIND} -n 64 -G 64 single_stream_gpuccl
    srun --cpu-bind=${CPU_BIND} -n 64 -G 64 single_stream_uniconn_gpuccl

    iter=$((iter + 1))
done
