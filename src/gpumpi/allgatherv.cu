#include "common.hpp"
#include "uniconn/gpumpi/coordinator.hpp"
namespace uniconn {

// template <typename T>
// GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::AllGatherv(T* sendbuf, T* recvbuf, size_t* counts,
//                                                                           size_t* displs,
//                                                                           Communicator<MPIBackend>* comm) {
//     if (temp_recv_counts.empty() && temp_recv_displs.empty()) {
//         for (size_t i = 0; i < comm->GlobalSize(); i++) {
//             temp_recv_counts.emplace_back(counts[i]);
//             temp_recv_displs.emplace_back(displs[i]);
//         }
//         if (is_grouped == 0) {
//             if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
//                 GPU_RT_CALL(UncGpuStreamSynchronize(stream));
//             }
//             MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Allgatherv)(internal::mpi::buf_or_inplace(sendbuf),
//                                                               counts[comm->GlobalRank()], internal::mpi::TypeMap<T>(),
//                                                               recvbuf, temp_recv_counts.data(), temp_recv_displs.data(),
//                                                               internal::mpi::TypeMap<T>(), comm->mpi_comm));
//             // for (size_t i = 0; i < comm->GlobalSize(); ++i) {
//             //     if (counts[i] > 0) {
//             //         MPI_CALL(MPI_Isend(sendbuf, counts[comm->GlobalRank()], MPI_DOUBLE, i, 2, comm->mpi_comm,
//             //                            &(internal_reqs.emplace_back(MPI_REQUEST_NULL))));
//             //         MPI_CALL(MPI_Irecv((recvbuf + displs[i]), counts[i], MPI_DOUBLE, i, 2, comm->mpi_comm,
//             //                            &(internal_reqs.emplace_back(MPI_REQUEST_NULL))));
//             //     }
//             // }
//             // MPI_CALL(MPI_Waitall(internal_reqs.size(), internal_reqs.data(), MPI_STATUSES_IGNORE));
//             // internal_reqs.clear();
//             temp_recv_counts.clear();
//             temp_recv_displs.clear();
//         } else {
//             MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Iallgatherv)(
//                 internal::mpi::buf_or_inplace(sendbuf), counts[comm->GlobalRank()], internal::mpi::TypeMap<T>(),
//                 recvbuf, temp_recv_counts.data(), temp_recv_displs.data(), internal::mpi::TypeMap<T>(), comm->mpi_comm,
//                 &internal_reqs.emplace_back(MPI_REQUEST_NULL)));
//         }
//     }
// }
// template <typename T>
// GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::AllGatherv(T* buffer, size_t* counts, size_t* displs,
//                                                                           Communicator<MPIBackend>* comm) {
//     AllGatherv(internal::IN_PLACE<T>, buffer, counts, displs, comm);
// }

// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::AllGatherv(T* sendbuf, T* recvbuf, size_t* counts,
//                                                                             size_t* displs,
//                                                                             Communicator<MPIBackend>* comm) {}
// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::AllGatherv(T* buffer, size_t* counts, size_t* displs,
//                                                                             Communicator<MPIBackend>* comm) {}

// #define DECL_FUNC(TYPE)                                                                                              \
//     template GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::AllGatherv<TYPE>(                        \
//         TYPE * sendbuf, TYPE * recvbuf, size_t* counts, size_t* displs, Communicator<MPIBackend>* comm);             \
//     template GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::AllGatherv<TYPE>(                        \
//         TYPE * buffer, size_t* counts, size_t* displs, Communicator<MPIBackend>* comm);                              \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::AllGatherv<ThreadGroup::BLOCK, TYPE>(  \
//         TYPE * sendbuf, TYPE * recvbuf, size_t* counts, size_t* displs, Communicator<MPIBackend>* comm);             \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::AllGatherv<ThreadGroup::BLOCK, TYPE>(  \
//         TYPE * buffer, size_t* counts, size_t* displs, Communicator<MPIBackend>* comm);                              \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::AllGatherv<ThreadGroup::WARP, TYPE>(   \
//         TYPE * sendbuf, TYPE * recvbuf, size_t* counts, size_t* displs, Communicator<MPIBackend>* comm);             \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::AllGatherv<ThreadGroup::WARP, TYPE>(   \
//         TYPE * buffer, size_t* counts, size_t* displs, Communicator<MPIBackend>* comm);                              \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::AllGatherv<ThreadGroup::THREAD, TYPE>( \
//         TYPE * sendbuf, TYPE * recvbuf, size_t* counts, size_t* displs, Communicator<MPIBackend>* comm);             \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::AllGatherv<ThreadGroup::THREAD, TYPE>( \
//         TYPE * buffer, size_t* counts, size_t* displs, Communicator<MPIBackend>* comm);

// UNC_GPUMPI_REPT(DECL_FUNC)
}  // namespace uniconn
