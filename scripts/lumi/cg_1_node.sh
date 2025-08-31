#!/bin/bash -l

#SBATCH --job-name=cg_1
#SBATCH --account={account_name}
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=0-04:00:00
#SBATCH --output=results/lumi/%x_%j.csv
#SBATCH --error=results/lumi/errors/%x-%j-stderr.txt

Uniconn_Dir=
Dataset_Dir=$Uniconn_Dir/datasets/
source $Uniconn_Dir/scripts/lumi/env.sh
cd $Uniconn_Dir/build/examples/CG-hip
max_gpu=8
iter_max=10

echo "benchmark, library, matrix_name, GPU_count, ms"
iter=0
while [ $iter -lt $iter_max ]; do
    for mtx in $Dataset_Dir/*.mtx; do
        gpu_num=1
        while [ $gpu_num -le $max_gpu ]; do
            srun --cpu-bind=${CPU_BIND} -n $gpu_num -G $gpu_num standard_mpi $mtx
            srun --cpu-bind=${CPU_BIND} -n $gpu_num -G $gpu_num standard_uniconn_mpi $mtx
            
            srun --cpu-bind=${CPU_BIND} -n $gpu_num -G $gpu_num standard_gpuccl $mtx
            srun --cpu-bind=${CPU_BIND} -n $gpu_num -G $gpu_num standard_uniconn_gpuccl $mtx
            gpu_num=$((gpu_num * 2))
        done
    done
    iter=$((iter + 1))
done
