
#include "common.hpp"
#include "uniconn/gpumpi/coordinator.hpp"
namespace uniconn {
// template <typename T>
// GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::AllGather(T* sendbuf, T* recvbuf, size_t count,
//                                                                          Communicator<MPIBackend>* comm) {
//     if (is_grouped == 0) {
//         if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
//             GPU_RT_CALL(UncGpuStreamSynchronize(stream));
//         }

//         MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Allgather)(internal::mpi::buf_or_inplace(sendbuf), count,
//                                                          internal::mpi::TypeMap<T>(), recvbuf, count,
//                                                          internal::mpi::TypeMap<T>(), comm->mpi_comm));

//     } else {
//         MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Iallgather)(
//             internal::mpi::buf_or_inplace(sendbuf), count, internal::mpi::TypeMap<T>(), recvbuf, count,
//             internal::mpi::TypeMap<T>(), comm->mpi_comm, &internal_reqs.emplace_back(MPI_REQUEST_NULL)));
//     }
// }

// template <typename T>
// GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::AllGather(T* buffer, size_t count,
//                                                                          Communicator<MPIBackend>* comm) {
//     AllGather(internal::IN_PLACE<T>, buffer, count, comm);
// }
// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::AllGather(T* sendbuf, T* recvbuf, size_t count,
//                                                                            Communicator<MPIBackend>* comm) {}
// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::AllGather(T* buffer, size_t count,
//                                                                            Communicator<MPIBackend>* comm) {}

// #define DECL_FUNC(TYPE)                                                                                             \
//     template GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::AllGather<TYPE>(                        \
//         TYPE * sendbuf, TYPE * recvbuf, size_t count, Communicator<MPIBackend>* comm);                              \
//     template GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::AllGather<TYPE>(                        \
//         TYPE * buffer, size_t count, Communicator<MPIBackend>* comm);                                               \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::AllGather<ThreadGroup::BLOCK, TYPE>(  \
//         TYPE * sendbuf, TYPE * recvbuf, size_t count, Communicator<MPIBackend>* comm);                              \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::AllGather<ThreadGroup::BLOCK, TYPE>(  \
//         TYPE * buffer, size_t count, Communicator<MPIBackend>* comm);                                               \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::AllGather<ThreadGroup::WARP, TYPE>(   \
//         TYPE * sendbuf, TYPE * recvbuf, size_t count, Communicator<MPIBackend>* comm);                              \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::AllGather<ThreadGroup::WARP, TYPE>(   \
//         TYPE * buffer, size_t count, Communicator<MPIBackend>* comm);                                               \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::AllGather<ThreadGroup::THREAD, TYPE>( \
//         TYPE * sendbuf, TYPE * recvbuf, size_t count, Communicator<MPIBackend>* comm);                              \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::AllGather<ThreadGroup::THREAD, TYPE>( \
//         TYPE * buffer, size_t count, Communicator<MPIBackend>* comm);

// UNC_GPUMPI_REPT(DECL_FUNC)

}  // namespace uniconn
