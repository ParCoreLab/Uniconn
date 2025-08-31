#!/bin/bash -l

#SBATCH --job-name=jacobi2D-004-nodes-0016-procs
#SBATCH --account={account_name}
#SBATCH --partition=acc
#SBATCH --qos={qos}
#SBATCH --nodes=4
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
iter_max=10

echo "benchmark, library, GPU_count, ms"
iter=0
while [ $iter -lt $iter_max ]; do
    mpirun -np 16 --map-by ppr:4:node  single_stream_mpi
    mpirun -np 16 --map-by ppr:4:node  single_stream_uniconn_mpi

    mpirun -np 16 --map-by ppr:4:node  single_stream_gpuccl
    mpirun -np 16 --map-by ppr:4:node  single_stream_uniconn_gpuccl

    mpirun -np 16 --map-by ppr:4:node  single_stream_nvshmem_host
    mpirun -np 16 --map-by ppr:4:node  single_stream_uniconn_nvshmem_host

    mpirun -np 16 --map-by ppr:4:node  single_stream_nvshmem_device
    mpirun -np 16 --map-by ppr:4:node  single_stream_uniconn_nvshmem_device
    iter=$((iter + 1))
done
