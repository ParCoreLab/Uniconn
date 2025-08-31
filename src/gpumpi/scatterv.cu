#include "common.hpp"
#include "uniconn/gpumpi/coordinator.hpp"
namespace uniconn {
// template <typename T>
// GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::Scatterv(const T* sendbuf, T* recvbuf, size_t* counts,
//                                                                         size_t* displs, int root,
//                                                                         Communicator<MPIBackend>* comm) {
//     if (sendbuf == internal::IN_PLACE<T> && comm->GlobalRank() == root) {
//         sendbuf = recvbuf;
//         recvbuf = internal::IN_PLACE<T>;
//     }
//     if (temp_send_counts.empty() && temp_send_displs.empty()) {
//         for (size_t i = 0; i < comm->GlobalSize(); i++) {
//             temp_send_counts.emplace_back(counts[i]);
//             temp_send_displs.emplace_back(displs[i]);
//         }
//         if (is_grouped == 0) {
//             if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
//                 GPU_RT_CALL(UncGpuStreamSynchronize(stream));
//             }
//             MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Scatterv)(
//                 sendbuf, temp_send_counts.data(), temp_send_displs.data(), internal::mpi::TypeMap<T>(),
//                 internal::mpi::buf_or_inplace(recvbuf), counts[comm->GlobalRank()], internal::mpi::TypeMap<T>(), root,
//                 comm->mpi_comm));
//             temp_send_counts.clear();
//             temp_send_displs.clear();
//         } else {
//             MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Iscatterv)(
//                 sendbuf, temp_send_counts.data(), temp_send_displs.data(), internal::mpi::TypeMap<T>(),
//                 internal::mpi::buf_or_inplace(recvbuf), counts[comm->GlobalRank()], internal::mpi::TypeMap<T>(), root,
//                 comm->mpi_comm, &internal_reqs.emplace_back(MPI_REQUEST_NULL)));
//         }
//     }
// }

// template <typename T>
// GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::Scatterv(T* buffer, size_t* counts, size_t* displs,
//                                                                         int root, Communicator<MPIBackend>* comm) {
//     Scatterv(internal::IN_PLACE<T>, buffer, counts, displs, root, comm);
// }

// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Scatterv(const T* sendbuf, T* recvbuf, size_t* counts,
//                                                                           size_t* displs, int root,
//                                                                           Communicator<MPIBackend>* comm) {}

// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Scatterv(T* buffer, size_t* counts, size_t* displs,
//                                                                           int root, Communicator<MPIBackend>* comm) {}

// #define DECL_FUNC(TYPE)                                                                                                \
//     template GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::Scatterv<TYPE>(                            \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t* counts, size_t* displs, int root, Communicator<MPIBackend>* comm); \
//     template GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::Scatterv<TYPE>(                            \
//         TYPE * buffer, size_t* counts, size_t* displs, int root, Communicator<MPIBackend>* comm);                      \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Scatterv<ThreadGroup::BLOCK, TYPE>(      \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t* counts, size_t* displs, int root, Communicator<MPIBackend>* comm); \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Scatterv<ThreadGroup::BLOCK, TYPE>(      \
//         TYPE * buffer, size_t* counts, size_t* displs, int root, Communicator<MPIBackend>* comm);                      \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Scatterv<ThreadGroup::WARP, TYPE>(       \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t* counts, size_t* displs, int root, Communicator<MPIBackend>* comm); \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Scatterv<ThreadGroup::WARP, TYPE>(       \
//         TYPE * buffer, size_t* counts, size_t* displs, int root, Communicator<MPIBackend>* comm);                      \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Scatterv<ThreadGroup::THREAD, TYPE>(     \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t* counts, size_t* displs, int root, Communicator<MPIBackend>* comm); \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Scatterv<ThreadGroup::THREAD, TYPE>(     \
//         TYPE * buffer, size_t* counts, size_t* displs, int root, Communicator<MPIBackend>* comm);

// UNC_GPUMPI_REPT(DECL_FUNC)

}  // namespace uniconn
