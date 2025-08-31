#include "common.hpp"
#include "uniconn/gpumpi/communicator.hpp"
namespace uniconn {
GPU_HOST Communicator<MPIBackend>::Communicator() { MPI_CALL(MPI_Comm_dup(MPI_COMM_WORLD, &(mpi_comm))); }

// GPU_HOST Communicator<MPIBackend>* Communicator<MPIBackend>::toDevice() { return nullptr; }

// GPU_UNIFIED int Communicator<MPIBackend>::GlobalSize() {
//     int ret = 0;
// #if !defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 0
//     MPI_CALL(MPI_Comm_size(mpi_comm, &ret));
// #endif
//     return ret;
// }

// GPU_UNIFIED int Communicator<MPIBackend>::GlobalRank() {
//     int ret = -1;
// #if !defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 0
//     MPI_CALL(MPI_Comm_rank(mpi_comm, &ret));
// #endif
//     return ret;
// }

// template <ThreadGroup SCOPE>
// GPU_DEVICE void Communicator<MPIBackend>::Barrier() {}

// GPU_HOST void Communicator<MPIBackend>::Barrier(UncGpuStream_t stream) {
//     if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
//         GPU_RT_CALL(UncGpuStreamSynchronize(stream));
//     }
//     MPI_CALL(MPI_Barrier(mpi_comm));
// }

GPU_HOST Communicator<MPIBackend>::~Communicator() { MPI_CALL(MPI_Comm_free(&(mpi_comm))); }

// template GPU_DEVICE void Communicator<MPIBackend>::Barrier<ThreadGroup::BLOCK>();
// template GPU_DEVICE void Communicator<MPIBackend>::Barrier<ThreadGroup::WARP>();
// template GPU_DEVICE void Communicator<MPIBackend>::Barrier<ThreadGroup::THREAD>();

}  // namespace uniconn
