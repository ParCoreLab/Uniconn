#include "common.hpp"
#include "uniconn/gpuccl/coordinator.hpp"
namespace uniconn {
// template <typename T>
// GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AlltoAll(const T* sendbuf, T* recvbuf, size_t count,
//                                                                            Communicator<GpucclBackend>* comm) {
//     NCCL_CALL(ncclGroupStart());
//     for (size_t i = 0; i < comm->GlobalSize(); ++i) {
//         NCCL_CALL(ncclSend((const void*)&sendbuf[i * count], count, internal::gpuccl::TypeMap<T>(), i, comm->nccl_comm,
//                            stream));
//         NCCL_CALL(
//             ncclRecv((void*)&recvbuf[i * count], count, internal::gpuccl::TypeMap<T>(), i, comm->nccl_comm, stream));
//     }
//     NCCL_CALL(ncclGroupEnd());
// }
// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AlltoAll(const T* sendbuf, T* recvbuf, size_t count,
//                                                                              Communicator<GpucclBackend>* comm) {}

// #define DECL_FUNC(TYPE)                                                                                               \
//     template GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AlltoAll<TYPE>(                        \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, Communicator<GpucclBackend>* comm);                         \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AlltoAll<ThreadGroup::BLOCK, TYPE>(  \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, Communicator<GpucclBackend>* comm);                         \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AlltoAll<ThreadGroup::WARP, TYPE>(   \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, Communicator<GpucclBackend>* comm);                         \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AlltoAll<ThreadGroup::THREAD, TYPE>( \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, Communicator<GpucclBackend>* comm);

// UNC_GPUCCL_REPT(DECL_FUNC)
}  // namespace uniconn
