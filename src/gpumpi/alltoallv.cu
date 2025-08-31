#include "common.hpp"
#include "uniconn/gpumpi/coordinator.hpp"
namespace uniconn {
// template <typename T>
// GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::AlltoAllv(const T* sendbuf, size_t* send_counts,
//                                                                          size_t* send_displs, T* recvbuf,
//                                                                          size_t* recv_counts, size_t* recv_displs,
//                                                                          Communicator<MPIBackend>* comm) {
//     if (temp_send_counts.empty() && temp_send_displs.empty() && temp_recv_counts.empty() && temp_recv_displs.empty()) {
//         for (size_t i = 0; i < comm->GlobalSize(); i++) {
//             temp_send_counts.emplace_back(send_counts[i]);
//             temp_send_displs.emplace_back(send_displs[i]);
//             temp_recv_counts.emplace_back(recv_counts[i]);
//             temp_recv_displs.emplace_back(recv_displs[i]);
//         }
//         if (is_grouped == 0) {
//             if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
//                 GPU_RT_CALL(UncGpuStreamSynchronize(stream));
//             }
//             MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Alltoallv)(
//                 sendbuf, temp_send_counts.data(), temp_send_displs.data(), internal::mpi::TypeMap<T>(), recvbuf,
//                 temp_recv_counts.data(), temp_recv_displs.data(), internal::mpi::TypeMap<T>(), comm->mpi_comm));
//             temp_send_counts.clear();
//             temp_send_displs.clear();
//             temp_recv_counts.clear();
//             temp_recv_displs.clear();
//         } else {
//             MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Ialltoallv)(
//                 sendbuf, temp_send_counts.data(), temp_send_displs.data(), internal::mpi::TypeMap<T>(), recvbuf,
//                 temp_recv_counts.data(), temp_recv_displs.data(), internal::mpi::TypeMap<T>(), comm->mpi_comm,
//                 &internal_reqs.emplace_back(MPI_REQUEST_NULL)));
//         }
//     }
// }

// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::AlltoAllv(const T* sendbuf, size_t* send_counts,
//                                                                            size_t* send_displs, T* recvbuf,
//                                                                            size_t* recv_counts, size_t* recv_displs,
//                                                                            Communicator<MPIBackend>* comm) {}
// #define DECL_FUNC(TYPE)                                                                                             \
//     template GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::AlltoAllv<TYPE>(                        \
//         const TYPE* sendbuf, size_t* send_counts, size_t* send_displs, TYPE* recvbuf, size_t* recv_counts,          \
//         size_t* recv_displs, Communicator<MPIBackend>* comm);                                                       \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::AlltoAllv<ThreadGroup::BLOCK, TYPE>(  \
//         const TYPE* sendbuf, size_t* send_counts, size_t* send_displs, TYPE* recvbuf, size_t* recv_counts,          \
//         size_t* recv_displs, Communicator<MPIBackend>* comm);                                                       \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::AlltoAllv<ThreadGroup::WARP, TYPE>(   \
//         const TYPE* sendbuf, size_t* send_counts, size_t* send_displs, TYPE* recvbuf, size_t* recv_counts,          \
//         size_t* recv_displs, Communicator<MPIBackend>* comm);                                                       \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::AlltoAllv<ThreadGroup::THREAD, TYPE>( \
//         const TYPE* sendbuf, size_t* send_counts, size_t* send_displs, TYPE* recvbuf, size_t* recv_counts,          \
//         size_t* recv_displs, Communicator<MPIBackend>* comm);

// UNC_GPUMPI_REPT(DECL_FUNC)
}  // namespace uniconn
