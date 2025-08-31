#include "common.hpp"
#include "uniconn/gpumpi/coordinator.hpp"
namespace uniconn {
// template <typename T>
// GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::AlltoAll(const T* sendbuf, T* recvbuf, size_t count,
//                                                                         Communicator<MPIBackend>* comm) {
//     if (is_grouped == 0) {
//         if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
//             GPU_RT_CALL(UncGpuStreamSynchronize(stream));
//         }
//         MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Alltoall)(sendbuf, count, internal::mpi::TypeMap<T>(), recvbuf,
//                                                         int(count), internal::mpi::TypeMap<T>(), comm->mpi_comm));
//     } else {
//         MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Ialltoall)(sendbuf, count, internal::mpi::TypeMap<T>(), recvbuf,
//                                                          int(count), internal::mpi::TypeMap<T>(), comm->mpi_comm,
//                                                          &internal_reqs.emplace_back(MPI_REQUEST_NULL)));
//     }
// }
// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::AlltoAll(const T* sendbuf, T* recvbuf, size_t count,
//                                                                           Communicator<MPIBackend>* comm) {}

// #define DECL_FUNC(TYPE)                                                                                            \
//     template GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::AlltoAll<TYPE>(                        \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, Communicator<MPIBackend>* comm);                         \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::AlltoAll<ThreadGroup::BLOCK, TYPE>(  \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, Communicator<MPIBackend>* comm);                         \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::AlltoAll<ThreadGroup::WARP, TYPE>(   \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, Communicator<MPIBackend>* comm);                         \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::AlltoAll<ThreadGroup::THREAD, TYPE>( \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, Communicator<MPIBackend>* comm);

// UNC_GPUMPI_REPT(DECL_FUNC)
}  // namespace uniconn
