#!/bin/bash -l

#SBATCH --job-name=CG-cuda-002-nodes-0008-procs
#SBATCH --account=ehpc114
#SBATCH --partition=acc
#SBATCH --qos={qos}
#SBATCH --nodes=2
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
iter=0
while [ $iter -lt $iter_max ]; do
    for mtx in $Dataset_Dir/*.mtx; do

        mpirun -np 8 --map-by ppr:4:node standard_mpi $mtx
        mpirun -np 8 --map-by ppr:4:node  standard_uniconn_mpi $mtx

        mpirun -np 8 --map-by ppr:4:node  standard_gpuccl $mtx
        mpirun -np 8 --map-by ppr:4:node  standard_uniconn_gpuccl $mtx

        mpirun -np 8 --map-by ppr:4:node  standard_nvshmem_host $mtx
        mpirun -np 8 --map-by ppr:4:node  standard_uniconn_nvshmem_host $mtx

        mpirun -np 8 --map-by ppr:4:node  standard_nvshmem_device $mtx
        mpirun -np 8 --map-by ppr:4:node  standard_uniconn_nvshmem_device $mtx

    done
    iter=$((iter + 1))
done
