#include "common.hpp"
#include "uniconn/gpuccl/coordinator.hpp"
namespace uniconn {
// template <typename T>
// GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Scatter(const T* sendbuf, T* recvbuf, size_t count,
//                                                                           int root, Communicator<GpucclBackend>* comm) {
//     NCCL_CALL(ncclGroupStart());

//     NCCL_CALL(ncclRecv((void*)recvbuf, count, internal::gpuccl::TypeMap<T>(), root, comm->nccl_comm, stream));
//     if (comm->GlobalRank() == root) {
//         for (size_t i = 0; i < comm->GlobalSize(); ++i) {
//             NCCL_CALL(ncclSend((const void*)&sendbuf[i * count], count, internal::gpuccl::TypeMap<T>(), i,
//                                comm->nccl_comm, stream));
//         }
//     }
//     NCCL_CALL(ncclGroupEnd());
// }
// template <typename T>
// GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Scatter(T* buffer, size_t count, int root,
//                                                                           Communicator<GpucclBackend>* comm) {
//     Scatter((buffer + comm->GlobalRank() * count), buffer, count, root, comm);
// }
// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Scatter(const T* sendbuf, T* recvbuf, size_t count,
//                                                                             int root,
//                                                                             Communicator<GpucclBackend>* comm) {}
// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Scatter(T* buffer, size_t count, int root,
//                                                                             Communicator<GpucclBackend>* comm) {}
// #define DECL_FUNC(TYPE)                                                                                              \
//     template GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Scatter<TYPE>(                        \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, int root, Communicator<GpucclBackend>* comm);              \
//     template GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Scatter<TYPE>(                        \
//         TYPE * buffer, size_t count, int root, Communicator<GpucclBackend>* comm);                                   \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Scatter<ThreadGroup::BLOCK, TYPE>(  \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, int root, Communicator<GpucclBackend>* comm);              \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Scatter<ThreadGroup::BLOCK, TYPE>(  \
//         TYPE * buffer, size_t count, int root, Communicator<GpucclBackend>* comm);                                   \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Scatter<ThreadGroup::WARP, TYPE>(   \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, int root, Communicator<GpucclBackend>* comm);              \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Scatter<ThreadGroup::WARP, TYPE>(   \
//         TYPE * buffer, size_t count, int root, Communicator<GpucclBackend>* comm);                                   \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Scatter<ThreadGroup::THREAD, TYPE>( \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, int root, Communicator<GpucclBackend>* comm);              \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Scatter<ThreadGroup::THREAD, TYPE>( \
//         TYPE * buffer, size_t count, int root, Communicator<GpucclBackend>* comm);

// UNC_GPUCCL_REPT(DECL_FUNC)

}  // namespace uniconn
