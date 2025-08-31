#include "common.hpp"
#include "uniconn/gpumpi/coordinator.hpp"
namespace uniconn {
// template <typename T>
// GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::Gather(const T* sendbuf, T* recvbuf, size_t count,
//                                                                       int root, Communicator<MPIBackend>* comm) {
//     if (is_grouped == 0) {
//         if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
//             GPU_RT_CALL(UncGpuStreamSynchronize(stream));
//         }
//         MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Gather)(internal::mpi::buf_or_inplace(sendbuf), count,
//                                                       internal::mpi::TypeMap<T>(), recvbuf, count,
//                                                       internal::mpi::TypeMap<T>(), root, comm->mpi_comm));
//     }else{
//         MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Igather)(internal::mpi::buf_or_inplace(sendbuf), count,
//                                                       internal::mpi::TypeMap<T>(), recvbuf, count,
//                                                       internal::mpi::TypeMap<T>(), root, comm->mpi_comm,&internal_reqs.emplace_back(MPI_REQUEST_NULL)));
//     }
// }
// template <typename T>
// GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::Gather(T* buffer, size_t count, int root,
//                                                                       Communicator<MPIBackend>* comm) {
//     Gather(internal::IN_PLACE<T>, buffer, count, root, comm);
// }
// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Gather(const T* sendbuf, T* recvbuf, size_t count,
//                                                                         int root, Communicator<MPIBackend>* comm) {}
// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Gather(T* buffer, size_t count, int root,
//                                                                         Communicator<MPIBackend>* comm) {}
// #define DECL_FUNC(TYPE)                                                                                          \
//     template GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::Gather<TYPE>(                        \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, int root, Communicator<MPIBackend>* comm);             \
//     template GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::Gather<TYPE>(                        \
//         TYPE * buffer, size_t count, int root, Communicator<MPIBackend>* comm);                                  \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Gather<ThreadGroup::BLOCK, TYPE>(  \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, int root, Communicator<MPIBackend>* comm);             \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Gather<ThreadGroup::BLOCK, TYPE>(  \
//         TYPE * buffer, size_t count, int root, Communicator<MPIBackend>* comm);                                  \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Gather<ThreadGroup::WARP, TYPE>(   \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, int root, Communicator<MPIBackend>* comm);             \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Gather<ThreadGroup::WARP, TYPE>(   \
//         TYPE * buffer, size_t count, int root, Communicator<MPIBackend>* comm);                                  \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Gather<ThreadGroup::THREAD, TYPE>( \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, int root, Communicator<MPIBackend>* comm);             \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Gather<ThreadGroup::THREAD, TYPE>( \
//         TYPE * buffer, size_t count, int root, Communicator<MPIBackend>* comm);

// UNC_GPUMPI_REPT(DECL_FUNC)

}  // namespace uniconn
