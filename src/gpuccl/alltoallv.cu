#include "common.hpp"
#include "uniconn/gpuccl/coordinator.hpp"
namespace uniconn {
// template <typename T>
// GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AlltoAllv(const T* sendbuf, size_t* send_counts,
//                                                                             size_t* send_displs, T* recvbuf,
//                                                                             size_t* recv_counts, size_t* recv_displs,
//                                                                             Communicator<GpucclBackend>* comm) {
//     NCCL_CALL(ncclGroupStart());
//     for (size_t i = 0; i < comm->GlobalSize(); ++i) {
//         if (recv_counts[i] > 0) {
//             NCCL_CALL(ncclRecv((void*)(recvbuf + recv_displs[i]), recv_counts[i], internal::gpuccl::TypeMap<T>(), i,
//                                comm->nccl_comm, stream));
//         }

//         if (send_counts[i] > 0) {
//             NCCL_CALL(ncclSend((const void*)(sendbuf + send_displs[i]), send_counts[i], internal::gpuccl::TypeMap<T>(),
//                                i, comm->nccl_comm, stream));
//         }
//     }
//     NCCL_CALL(ncclGroupEnd());
// }
// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AlltoAllv(const T* sendbuf, size_t* send_counts,
//                                                                               size_t* send_displs, T* recvbuf,
//                                                                               size_t* recv_counts, size_t* recv_displs,
//                                                                               Communicator<GpucclBackend>* comm) {}
// #define DECL_FUNC(TYPE)                                                                                                \
//     template GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AlltoAllv<TYPE>(                        \
//         const TYPE* sendbuf, size_t* send_counts, size_t* send_displs, TYPE* recvbuf, size_t* recv_counts,             \
//         size_t* recv_displs, Communicator<GpucclBackend>* comm);                                                       \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AlltoAllv<ThreadGroup::BLOCK, TYPE>(  \
//         const TYPE* sendbuf, size_t* send_counts, size_t* send_displs, TYPE* recvbuf, size_t* recv_counts,             \
//         size_t* recv_displs, Communicator<GpucclBackend>* comm);                                                       \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AlltoAllv<ThreadGroup::WARP, TYPE>(   \
//         const TYPE* sendbuf, size_t* send_counts, size_t* send_displs, TYPE* recvbuf, size_t* recv_counts,             \
//         size_t* recv_displs, Communicator<GpucclBackend>* comm);                                                       \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AlltoAllv<ThreadGroup::THREAD, TYPE>( \
//         const TYPE* sendbuf, size_t* send_counts, size_t* send_displs, TYPE* recvbuf, size_t* recv_counts,             \
//         size_t* recv_displs, Communicator<GpucclBackend>* comm);

// UNC_GPUCCL_REPT(DECL_FUNC)
}  // namespace uniconn
