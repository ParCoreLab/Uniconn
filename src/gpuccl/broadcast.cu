#include "uniconn/gpuccl/coordinator.hpp"
#include "common.hpp"
namespace uniconn {
// template <typename T>
// GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Broadcast(T* buffer, size_t count, int root,
//                                                                             Communicator<GpucclBackend>* comm) {
    
//     NCCL_CALL(ncclBroadcast((const void*)buffer, (void*)buffer, count, internal::gpuccl::TypeMap<T>(), root,
//                             comm->nccl_comm, stream));
// }
// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Broadcast(T* buffer, size_t count, int root,
//                                                                               Communicator<GpucclBackend>* comm) {}

// #define DECL_FUNC(TYPE)                                                                                                \
//     template GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Broadcast<TYPE>(                        \
//         TYPE * buffer, size_t count, int root, Communicator<GpucclBackend>* comm);                                     \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Broadcast<ThreadGroup::BLOCK, TYPE>(  \
//         TYPE * buffer, size_t count, int root, Communicator<GpucclBackend>* comm);                                     \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Broadcast<ThreadGroup::WARP, TYPE>(   \
//         TYPE * buffer, size_t count, int root, Communicator<GpucclBackend>* comm);                                     \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Broadcast<ThreadGroup::THREAD, TYPE>( \
//         TYPE * buffer, size_t count, int root, Communicator<GpucclBackend>* comm);

// UNC_GPUCCL_REPT(DECL_FUNC)

}  // namespace uniconn
