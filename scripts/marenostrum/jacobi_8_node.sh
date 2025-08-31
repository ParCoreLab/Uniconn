#!/bin/bash -l

#SBATCH --job-name=jacobi2D-008-nodes-0032-procs
#SBATCH --account={account_name}
#SBATCH --partition=acc
#SBATCH --qos={qos}
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=0-03:59:00
#SBATCH --output=results/marenostrum/%x_%j.csv
#SBATCH --error=results/marenostrum/errors/%x-%j-stderr.txt

Uniconn_Dir=
source $Uniconn_Dir/scripts/marenostrum/env.sh
cd $Uniconn_Dir/build/examples/jacobi2D
iter_max=10
export NVSHMEM_DISABLE_NCCL=true
echo "benchmark, library, GPU_count, ms"
iter=0
while [ $iter -lt $iter_max ]; do
    mpirun  -np 32 --map-by numa -rank-by numa --bind-to numa single_stream_mpi
    mpirun  -np 32 --map-by numa -rank-by numa --bind-to numa single_stream_uniconn_mpi

    mpirun  -np 32 --map-by numa -rank-by numa --bind-to numa single_stream_gpuccl
    mpirun  -np 32 --map-by numa -rank-by numa --bind-to numa single_stream_uniconn_gpuccl

    mpirun -np 32 --map-by numa -rank-by numa --bind-to numa single_stream_nvshmem_host
    mpirun -np 32 --map-by numa -rank-by numa --bind-to numa single_stream_uniconn_nvshmem_host

    mpirun -np 32 --map-by numa -rank-by numa --bind-to numa single_stream_nvshmem_device
    mpirun -np 32 --map-by numa -rank-by numa --bind-to numa single_stream_uniconn_nvshmem_device
    iter=$((iter + 1))
done
