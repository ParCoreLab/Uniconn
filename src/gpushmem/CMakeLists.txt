set_source_path(THIS_DIR_CUDA_SOURCES
allgather.cu
allgatherv.cu
allreduce_and.cu
allreduce_or.cu
allreduce_xor.cu
allreduce_min.cu
allreduce_max.cu
allreduce_sum.cu
allreduce_prod.cu
alltoall.cu
alltoallv.cu
broadcast.cu
communicator.cu
coordinator.cu
environment.cu
gather.cu
gatherv.cu
memory.cu
reduce_and.cu
reduce_or.cu
reduce_xor.cu
reduce_min.cu
reduce_max.cu
reduce_sum.cu
reduce_prod.cu
scatter.cu
scatterv.cu
peertopeer.cu
)

# Propagate the files up the tree
set(CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CUDA_SOURCES}" PARENT_SCOPE)
