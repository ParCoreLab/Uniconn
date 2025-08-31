
#include "uniconn/gpuccl/coordinator.hpp"
#include "common.hpp"
namespace uniconn {
// template <typename T>
// GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllGather(T* sendbuf, T* recvbuf, size_t count,
//                                                                             Communicator<GpucclBackend>* comm) {
    
//     NCCL_CALL(ncclAllGather((const void*)sendbuf, (void*)recvbuf, count, internal::gpuccl::TypeMap<T>(),
//                             comm->nccl_comm, stream));
// }

// template <typename T>
// GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllGather(T* buffer, size_t count,
//                                                                             Communicator<GpucclBackend>* comm) {
//     AllGather((buffer + comm->GlobalRank() * count), buffer, count, comm);
// }
// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllGather(T* sendbuf, T* recvbuf, size_t count,
//                                                                               Communicator<GpucclBackend>* comm) {}
// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllGather(T* buffer, size_t count,
//                                                                               Communicator<GpucclBackend>* comm) {}

// #define DECL_FUNC(TYPE)                                                                                                \
//     template GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllGather<TYPE>(                        \
//         TYPE * sendbuf, TYPE * recvbuf, size_t count, Communicator<GpucclBackend>* comm);                              \
//     template GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllGather<TYPE>(                        \
//         TYPE * buffer, size_t count, Communicator<GpucclBackend>* comm);                                               \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllGather<ThreadGroup::BLOCK, TYPE>(  \
//         TYPE * sendbuf, TYPE * recvbuf, size_t count, Communicator<GpucclBackend>* comm);                              \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllGather<ThreadGroup::BLOCK, TYPE>(  \
//         TYPE * buffer, size_t count, Communicator<GpucclBackend>* comm);                                               \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllGather<ThreadGroup::WARP, TYPE>(   \
//         TYPE * sendbuf, TYPE * recvbuf, size_t count, Communicator<GpucclBackend>* comm);                              \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllGather<ThreadGroup::WARP, TYPE>(   \
//         TYPE * buffer, size_t count, Communicator<GpucclBackend>* comm);                                               \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllGather<ThreadGroup::THREAD, TYPE>( \
//         TYPE * sendbuf, TYPE * recvbuf, size_t count, Communicator<GpucclBackend>* comm);                              \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllGather<ThreadGroup::THREAD, TYPE>( \
//         TYPE * buffer, size_t count, Communicator<GpucclBackend>* comm);

// UNC_GPUCCL_REPT(DECL_FUNC)

}  // namespace uniconn
