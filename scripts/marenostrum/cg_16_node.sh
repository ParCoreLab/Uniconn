#!/bin/bash -l

#SBATCH --job-name=uniconn_CG-016-nodes-0064-procs
#SBATCH --account={account_name}
#SBATCH --partition=acc
#SBATCH --qos={qos}
#SBATCH --nodes=16
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-node=4
#SBATCH --time=0-03:59:00
#SBATCH --output=results/marenostrum/%x_%j.csv
#SBATCH --error=results/marenostrum/errors/%x-%j-stderr.txt

Uniconn_Dir=
Dataset_Dir=$Uniconn_Dir/datasets/
source $Uniconn_Dir/scripts/marenostrum/env.sh
cd $Uniconn_Dir/build/examples/CG-cuda

iter_max=10

echo "benchmark, library, matrix_name, GPU_count, ms"
for mtx in $Dataset_Dir/*.mtx; do
    iter=0
    while [ $iter -lt $iter_max ]; do
        mpirun -np 64 --map-by ppr:4:node standard_mpi $mtx
        mpirun -np 64 --map-by ppr:4:node standard_uniconn_mpi $mtx
        
        mpirun -np 64 --map-by ppr:4:node standard_gpuccl $mtx
        mpirun -np 64 --map-by ppr:4:node standard_uniconn_gpuccl $mtx
        
        mpirun -np 64 --map-by ppr:4:node standard_nvshmem_host $mtx
        mpirun -np 64 --map-by ppr:4:node standard_uniconn_nvshmem_host $mtx
        
        mpirun -np 64 --map-by ppr:4:node standard_nvshmem_device $mtx
        mpirun -np 64 --map-by ppr:4:node standard_uniconn_nvshmem_device $mtx
        iter=$((iter + 1))
    done
done
