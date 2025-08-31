#!/bin/bash -l

#SBATCH --job-name=bandwidth-001-nodes-0002-procs
#SBATCH --account={account_name}
#SBATCH --partition=acc
#SBATCH --qos={qos}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=80
#SBATCH --threads-per-core=2
#SBATCH --gpus-per-node=2
#SBATCH --time=0-01:00:00
#SBATCH --output=results/marenostrum/%x_%j.csv
#SBATCH --error=results/marenostrum/errors/%x-%j-stderr.txt

Uniconn_Dir=
source $Uniconn_Dir/scripts/marenostrum/env.sh
cd $Uniconn_Dir/build/examples/benchmarks

iter_max=10

echo "benchmark, library, buf_size, MB/s"
iter=0
while [ $iter -lt $iter_max ]; do
    mpirun -np 2 --map-by ppr:2:node bandwidth_mpi
    mpirun -np 2 --map-by ppr:2:node bandwidth_uniconn_mpi

    mpirun -np 2 --map-by ppr:2:node bandwidth_gpuccl
    mpirun -np 2 --map-by ppr:2:node bandwidth_uniconn_gpuccl

    mpirun -np 2 --map-by ppr:2:node bandwidth_nvshmem_device
    mpirun -np 2 --map-by ppr:2:node bandwidth_uniconn_nvshmem_device
    iter=$((iter + 1))
done
