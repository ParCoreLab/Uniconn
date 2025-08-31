#!/bin/bash -l
#SBATCH --job-name=cg_2_half
#SBATCH --account={account_name}
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=0-02:00:00
#SBATCH --output=results/lumi/%x_%j.csv
#SBATCH --error=results/lumi/errors/%x-%j-stderr.txt

Uniconn_Dir=
Dataset_Dir=$Uniconn_Dir/datasets/
source $Uniconn_Dir/scripts/lumi/env.sh
cd $Uniconn_Dir/build/examples/CG-hip
iter_max=10

echo "benchmark, library, matrix_name, GPU_count, ms"
iter=0
while [ $iter -lt $iter_max ]; do
    for mtx in $Dataset_Dir/*.mtx; do

        srun --cpu-bind=${CPU_BIND} -n 8 -G 8 --ntasks-per-node=4 --gpus-per-node=4 standard_mpi $mtx
        srun --cpu-bind=${CPU_BIND} -n 8 -G 8 --ntasks-per-node=4 --gpus-per-node=4 standard_uniconn_mpi $mtx

        srun --cpu-bind=${CPU_BIND} -n 8 -G 8 --ntasks-per-node=4 --gpus-per-node=4 standard_gpuccl $mtx
        srun --cpu-bind=${CPU_BIND} -n 8 -G 8 --ntasks-per-node=4 --gpus-per-node=4 standard_uniconn_gpuccl $mtx
    done
    iter=$((iter + 1))
done
