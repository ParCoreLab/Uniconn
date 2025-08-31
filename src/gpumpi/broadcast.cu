#include "common.hpp"
#include "uniconn/gpumpi/coordinator.hpp"
namespace uniconn {
// template <typename T>
// GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::Broadcast(T* buffer, size_t count, int root,
//                                                                          Communicator<MPIBackend>* comm) {
//     if (is_grouped == 0) {
//         if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
//             GPU_RT_CALL(UncGpuStreamSynchronize(stream));
//         }
//         MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Bcast)(buffer, count, internal::mpi::TypeMap<T>(), root, comm->mpi_comm));
//     } else {
//         MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Ibcast)(buffer, count, internal::mpi::TypeMap<T>(), root, comm->mpi_comm,
//                                                       &internal_reqs.emplace_back(MPI_REQUEST_NULL)));
//     }
// }
// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Broadcast(T* buffer, size_t count, int root,
//                                                                            Communicator<MPIBackend>* comm) {}

// #define DECL_FUNC(TYPE)                                                                                             \
//     template GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::Broadcast<TYPE>(                        \
//         TYPE * buffer, size_t count, int root, Communicator<MPIBackend>* comm);                                     \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Broadcast<ThreadGroup::BLOCK, TYPE>(  \
//         TYPE * buffer, size_t count, int root, Communicator<MPIBackend>* comm);                                     \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Broadcast<ThreadGroup::WARP, TYPE>(   \
//         TYPE * buffer, size_t count, int root, Communicator<MPIBackend>* comm);                                     \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Broadcast<ThreadGroup::THREAD, TYPE>( \
//         TYPE * buffer, size_t count, int root, Communicator<MPIBackend>* comm);

// UNC_GPUMPI_REPT(DECL_FUNC)

}  // namespace uniconn
