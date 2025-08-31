#include "common.hpp"
#include "uniconn/gpuccl/coordinator.hpp"
namespace uniconn {

// template <typename T>
// GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllGatherv(T* sendbuf, T* recvbuf, size_t* counts,
//                                                                              size_t* displs,
//                                                                              Communicator<GpucclBackend>* comm) {
//     NCCL_CALL(ncclGroupStart());
//     for (size_t i = 0; i < comm->GlobalSize(); ++i) {
//         if (counts[i] > 0) {
//             NCCL_CALL(ncclSend((const void*)sendbuf, counts[comm->GlobalRank()], internal::gpuccl::TypeMap<T>(), i,
//                                comm->nccl_comm, stream));
//             NCCL_CALL(ncclRecv((void*)(recvbuf + displs[i]), counts[i], internal::gpuccl::TypeMap<T>(), i,
//                                comm->nccl_comm, stream));
//         }
//     }
//     NCCL_CALL(ncclGroupEnd());
// }
// template <typename T>
// GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllGatherv(T* buffer, size_t* counts, size_t* displs,
//                                                                              Communicator<GpucclBackend>* comm) {
//     AllGatherv((buffer + displs[comm->GlobalRank()]), buffer, counts, displs, comm);
// }

// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllGatherv(T* sendbuf, T* recvbuf, size_t* counts,
//                                                                                size_t* displs,
//                                                                                Communicator<GpucclBackend>* comm) {}
// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllGatherv(T* buffer, size_t* counts,
//                                                                                size_t* displs,
//                                                                                Communicator<GpucclBackend>* comm) {}

// #define DECL_FUNC(TYPE)                                                                                                \
//     template GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllGatherv<TYPE>(                       \
//         TYPE * sendbuf, TYPE * recvbuf, size_t* counts, size_t* displs, Communicator<GpucclBackend>* comm);            \
//     template GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllGatherv<TYPE>(                       \
//         TYPE * buffer, size_t* counts, size_t* displs, Communicator<GpucclBackend>* comm);                             \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllGatherv<ThreadGroup::BLOCK, TYPE>( \
//         TYPE * sendbuf, TYPE * recvbuf, size_t* counts, size_t* displs, Communicator<GpucclBackend>* comm);            \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllGatherv<ThreadGroup::BLOCK, TYPE>( \
//         TYPE * buffer, size_t* counts, size_t* displs, Communicator<GpucclBackend>* comm);                             \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllGatherv<ThreadGroup::WARP, TYPE>(  \
//         TYPE * sendbuf, TYPE * recvbuf, size_t* counts, size_t* displs, Communicator<GpucclBackend>* comm);            \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllGatherv<ThreadGroup::WARP, TYPE>(  \
//         TYPE * buffer, size_t* counts, size_t* displs, Communicator<GpucclBackend>* comm);                             \
//     template GPU_DEVICE void                                                                                           \
//     Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllGatherv<ThreadGroup::THREAD, TYPE>(                         \
//         TYPE * sendbuf, TYPE * recvbuf, size_t* counts, size_t* displs, Communicator<GpucclBackend>* comm);            \
//     template GPU_DEVICE void                                                                                           \
//     Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllGatherv<ThreadGroup::THREAD, TYPE>(                         \
//         TYPE * buffer, size_t* counts, size_t* displs, Communicator<GpucclBackend>* comm);

// UNC_GPUCCL_REPT(DECL_FUNC)
}  // namespace uniconn
