
#include "common.hpp"
#include "uniconn/gpuccl/communicator.hpp"
namespace uniconn {

GPU_HOST Communicator<GpucclBackend>::Communicator() {
    ncclUniqueId nccl_uid;
    int rank = -1;
    int size = 0;
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));
    if (rank == 0) {
        NCCL_CALL(ncclGetUniqueId(&nccl_uid));
    }
    MPI_CALL(MPI_Bcast(&nccl_uid, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));
    NCCL_CALL(ncclCommInitRank(&nccl_comm, size, nccl_uid, rank));
    barrier_buf = Memory<GpucclBackend>::Alloc<unsigned char>(1);
}

// GPU_HOST Communicator<GpucclBackend>* Communicator<GpucclBackend>::toDevice() { return nullptr; }

// GPU_UNIFIED int Communicator<GpucclBackend>::GlobalSize() {
//     int ret = 0;
// #if !defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 0
//     NCCL_CALL(ncclCommCount(nccl_comm, &ret));
// #endif
//     return ret;
// }

// GPU_UNIFIED int Communicator<GpucclBackend>::GlobalRank() {
//     int ret = -1;
// #if !defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 0
//     NCCL_CALL(ncclCommUserRank(nccl_comm, &ret));
// #endif
//     return ret;
// }

// template <ThreadGroup SCOPE>
// GPU_DEVICE void Communicator<GpucclBackend>::Barrier() {}

// GPU_HOST void Communicator<GpucclBackend>::Barrier(UncGpuStream_t stream) {
//     NCCL_CALL(ncclAllReduce((const void*)barrier_buf, (void*)barrier_buf, 1, internal::gpuccl::TypeMap<unsigned char>(),
//                             ncclSum, nccl_comm, stream));
// }

GPU_HOST Communicator<GpucclBackend>::~Communicator() {
    Memory<GpucclBackend>::Free(barrier_buf);
    NCCL_CALL(ncclCommDestroy(nccl_comm));
}

// template GPU_DEVICE void Communicator<GpucclBackend>::Barrier<ThreadGroup::BLOCK>();
// template GPU_DEVICE void Communicator<GpucclBackend>::Barrier<ThreadGroup::WARP>();
// template GPU_DEVICE void Communicator<GpucclBackend>::Barrier<ThreadGroup::THREAD>();

}  // namespace uniconn
