#include "common.hpp"
#include "uniconn/gpuccl/coordinator.hpp"
namespace uniconn {
// template <typename T>
// GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Gather(const T* sendbuf, T* recvbuf, size_t count,
//                                                                          int root, Communicator<GpucclBackend>* comm) {
//     NCCL_CALL(ncclGroupStart());
//     NCCL_CALL(ncclSend((const void*)sendbuf, count, internal::gpuccl::TypeMap<T>(), root, comm->nccl_comm, stream));
//     if (comm->GlobalRank() == root) {
//         for (size_t i = 0; i < comm->GlobalSize(); ++i) {
//             NCCL_CALL(ncclRecv((void*)&recvbuf[i * count], count, internal::gpuccl::TypeMap<T>(), i, comm->nccl_comm,
//                                stream));
//         }
//     }
//     NCCL_CALL(ncclGroupEnd());
// }
// template <typename T>
// GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Gather(T* buffer, size_t count, int root,
//                                                                          Communicator<GpucclBackend>* comm) {
//     Gather((buffer + comm->GlobalRank() * count), buffer, count, root, comm);
// }
// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Gather(const T* sendbuf, T* recvbuf, size_t count,
//                                                                            int root,
//                                                                            Communicator<GpucclBackend>* comm) {}
// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Gather(T* buffer, size_t count, int root,
//                                                                            Communicator<GpucclBackend>* comm) {}
// #define DECL_FUNC(TYPE)                                                                                             \
//     template GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Gather<TYPE>(                        \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, int root, Communicator<GpucclBackend>* comm);             \
//     template GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Gather<TYPE>(                        \
//         TYPE * buffer, size_t count, int root, Communicator<GpucclBackend>* comm);                                  \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Gather<ThreadGroup::BLOCK, TYPE>(  \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, int root, Communicator<GpucclBackend>* comm);             \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Gather<ThreadGroup::BLOCK, TYPE>(  \
//         TYPE * buffer, size_t count, int root, Communicator<GpucclBackend>* comm);                                  \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Gather<ThreadGroup::WARP, TYPE>(   \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, int root, Communicator<GpucclBackend>* comm);             \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Gather<ThreadGroup::WARP, TYPE>(   \
//         TYPE * buffer, size_t count, int root, Communicator<GpucclBackend>* comm);                                  \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Gather<ThreadGroup::THREAD, TYPE>( \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, int root, Communicator<GpucclBackend>* comm);             \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Gather<ThreadGroup::THREAD, TYPE>( \
//         TYPE * buffer, size_t count, int root, Communicator<GpucclBackend>* comm);

// UNC_GPUCCL_REPT(DECL_FUNC)

}  // namespace uniconn
