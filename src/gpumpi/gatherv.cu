#include "common.hpp"
#include "uniconn/gpumpi/coordinator.hpp"
namespace uniconn {
// template <typename T>
// GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::Gatherv(const T* sendbuf, T* recvbuf, size_t* counts,
//                                                                        size_t* displs, int root,
//                                                                        Communicator<MPIBackend>* comm) {
//     if (sendbuf == internal::IN_PLACE<T> && comm->GlobalRank() != root) {
//         sendbuf = recvbuf;
//     }
//     if (temp_recv_counts.empty() && temp_recv_displs.empty()) {
//         for (size_t i = 0; i < comm->GlobalSize(); i++) {
//             temp_recv_counts.emplace_back(counts[i]);
//             temp_recv_displs.emplace_back(displs[i]);
//         }
//         if (is_grouped == 0) {
//             if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
//                 GPU_RT_CALL(UncGpuStreamSynchronize(stream));
//             }
//             MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Gatherv)(internal::mpi::buf_or_inplace(sendbuf),
//                                                            counts[comm->GlobalRank()], internal::mpi::TypeMap<T>(),
//                                                            recvbuf, temp_recv_counts.data(), temp_recv_displs.data(),
//                                                            internal::mpi::TypeMap<T>(), root, comm->mpi_comm));
//             temp_recv_counts.clear();
//             temp_recv_displs.clear();
//         } else {
//             MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Igatherv)(
//                 internal::mpi::buf_or_inplace(sendbuf), counts[comm->GlobalRank()], internal::mpi::TypeMap<T>(),
//                 recvbuf, temp_recv_counts.data(), temp_recv_displs.data(), internal::mpi::TypeMap<T>(), root,
//                 comm->mpi_comm, &internal_reqs.emplace_back(MPI_REQUEST_NULL)));
//         }
//     }
// }
// template <typename T>
// GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::Gatherv(T* buffer, size_t* counts, size_t* displs,
//                                                                        int root, Communicator<MPIBackend>* comm) {
//     Gatherv(internal::IN_PLACE<T>, buffer, counts, displs, root, comm);
// }
// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Gatherv(const T* sendbuf, T* recvbuf, size_t* counts,
//                                                                          size_t* displs, int root,
//                                                                          Communicator<MPIBackend>* comm) {}
// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Gatherv(T* buffer, size_t* counts, size_t* displs,
//                                                                          int root, Communicator<MPIBackend>* comm) {}
// #define DECL_FUNC(TYPE)                                                                                                \
//     template GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::Gatherv<TYPE>(                             \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t* counts, size_t* displs, int root, Communicator<MPIBackend>* comm); \
//     template GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::Gatherv<TYPE>(                             \
//         TYPE * buffer, size_t* counts, size_t* displs, int root, Communicator<MPIBackend>* comm);                      \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Gatherv<ThreadGroup::BLOCK, TYPE>(       \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t* counts, size_t* displs, int root, Communicator<MPIBackend>* comm); \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Gatherv<ThreadGroup::BLOCK, TYPE>(       \
//         TYPE * buffer, size_t* counts, size_t* displs, int root, Communicator<MPIBackend>* comm);                      \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Gatherv<ThreadGroup::WARP, TYPE>(        \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t* counts, size_t* displs, int root, Communicator<MPIBackend>* comm); \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Gatherv<ThreadGroup::WARP, TYPE>(        \
//         TYPE * buffer, size_t* counts, size_t* displs, int root, Communicator<MPIBackend>* comm);                      \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Gatherv<ThreadGroup::THREAD, TYPE>(      \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t* counts, size_t* displs, int root, Communicator<MPIBackend>* comm); \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Gatherv<ThreadGroup::THREAD, TYPE>(      \
//         TYPE * buffer, size_t* counts, size_t* displs, int root, Communicator<MPIBackend>* comm);

// UNC_GPUMPI_REPT(DECL_FUNC)
}  // namespace uniconn
