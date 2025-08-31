set_source_path(THIS_DIR_CUDA_SOURCES
allgather.cu
allgatherv.cu
allreduce.cu
alltoall.cu
alltoallv.cu
broadcast.cu
#common.hpp
communicator.cu
coordinator.cu
environment.cu
gather.cu
gatherv.cu
memory.cu
peertopeer.cu
reduce.cu
scatter.cu
scatterv.cu
)

# Propagate the files up the tree
set(CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CUDA_SOURCES}" PARENT_SCOPE)
