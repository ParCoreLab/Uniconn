#!/bin/bash -l

#SBATCH --job-name=jacobi2D-001-nodes-0004-procs
#SBATCH --account={account_name}
#SBATCH --partition=acc
#SBATCH --qos={qos}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=40
#SBATCH --threads-per-core=2
#SBATCH --gpus-per-node=4
#SBATCH --time=0-03:59:00
#SBATCH --output=results/marenostrum/%x_%j.csv
#SBATCH --error=results/marenostrum/errors/%x-%j-stderr.txt

Uniconn_Dir=
source $Uniconn_Dir/scripts/marenostrum/env.sh
cd $Uniconn_Dir/build/examples/jacobi2D
max_gpu=4
iter_max=10

echo "benchmark, library, GPU_count, ms"
iter=0
while [ $iter -lt $iter_max ]; do
    gpu_num=1
    while [ $gpu_num -le $max_gpu ]; do
        mpirun -np $gpu_num --map-by ppr:$gpu_num:node  single_stream_mpi
        mpirun -np $gpu_num --map-by ppr:$gpu_num:node  single_stream_uniconn_mpi

        mpirun -np $gpu_num --map-by ppr:$gpu_num:node  single_stream_gpuccl
        mpirun -np $gpu_num --map-by ppr:$gpu_num:node  single_stream_uniconn_gpuccl

        mpirun -np $gpu_num --map-by ppr:$gpu_num:node  single_stream_nvshmem_host
        mpirun -np $gpu_num --map-by ppr:$gpu_num:node  single_stream_uniconn_nvshmem_host

        mpirun -np $gpu_num --map-by ppr:$gpu_num:node  single_stream_nvshmem_device
        mpirun -np $gpu_num --map-by ppr:$gpu_num:node  single_stream_uniconn_nvshmem_device
        gpu_num=$((gpu_num * 2))
    done
    iter=$((iter + 1))
done
