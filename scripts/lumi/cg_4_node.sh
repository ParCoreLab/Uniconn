#!/bin/bash -l
#SBATCH --job-name=cg_4
#SBATCH --account={account_name}
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=0-01:00:00
#SBATCH --output=results/lumi/%x_%j.csv
#SBATCH --error=results/lumi/errors/%x-%j-stderr.txt

Uniconn_Dir=
Dataset_Dir=$Uniconn_Dir/datasets/
source $Uniconn_Dir/scripts/lumi/env.sh
cd $Uniconn_Dir/build/examples/CG-hip
iter_max=10

echo "benchmark, library, matrix_name, GPU_count, ms"
for mtx in $Dataset_Dir/*.mtx; do
    iter=0
    while [ $iter -lt $iter_max ]; do
        srun --cpu-bind=${CPU_BIND} -n 32 -G 32 standard_mpi $mtx
        srun --cpu-bind=${CPU_BIND} -n 32 -G 32 standard_uniconn_mpi $mtx
        
        srun --cpu-bind=${CPU_BIND} -n 32 -G 32 standard_gpuccl $mtx
        srun --cpu-bind=${CPU_BIND} -n 32 -G 32 standard_uniconn_gpuccl $mtx
        iter=$((iter + 1))
    done
done
