#include "uniconn/gpuccl/coordinator.hpp"
#include "common.hpp"
namespace uniconn {
// template <ReductionOperator OP, typename T>
// GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Reduce(const T* sendbuf, T* recvbuf, size_t count,
//                                                                          int root, Communicator<GpucclBackend>* comm) {
    
//     NCCL_CALL(ncclReduce((const void*)sendbuf, (void*)recvbuf, count, internal::gpuccl::TypeMap<T>(),
//                          internal::gpuccl::ReductOp2ncclRedOp<OP>(), root, comm->nccl_comm, stream));
// }
// template <ReductionOperator OP, typename T>
// GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Reduce(T* buffer, size_t count, int root,
//                                                                          Communicator<GpucclBackend>* comm) {
//     Reduce<OP>(buffer, buffer, count, root, comm);
// }

// template <ThreadGroup SCOPE, ReductionOperator OP, typename T>
// GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Reduce(const T* sendbuf, T* recvbuf, size_t count,
//                                                                            int root,
//                                                                            Communicator<GpucclBackend>* comm) {}
// template <ThreadGroup SCOPE, ReductionOperator OP, typename T>
// GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Reduce(T* buffer, size_t count, int root,
//                                                                            Communicator<GpucclBackend>* comm) {}

// #define DECL_FUNC(TYPE, OP)                                                                                         \
//     template GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Reduce<ReductionOperator::OP, TYPE>( \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, int root, Communicator<GpucclBackend>* comm);             \
//     template GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Reduce<ReductionOperator::OP, TYPE>( \
//         TYPE * buffer, size_t count, int root, Communicator<GpucclBackend>* comm);                                  \
//     template GPU_DEVICE void                                                                                        \
//     Coordinator<GpucclBackend, LaunchMode::HostDriven>::Reduce<ThreadGroup::BLOCK, ReductionOperator::OP, TYPE>(    \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, int root, Communicator<GpucclBackend>* comm);             \
//     template GPU_DEVICE void                                                                                        \
//     Coordinator<GpucclBackend, LaunchMode::HostDriven>::Reduce<ThreadGroup::BLOCK, ReductionOperator::OP, TYPE>(    \
//         TYPE * buffer, size_t count, int root, Communicator<GpucclBackend>* comm);                                  \
//     template GPU_DEVICE void                                                                                        \
//     Coordinator<GpucclBackend, LaunchMode::HostDriven>::Reduce<ThreadGroup::WARP, ReductionOperator::OP, TYPE>(     \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, int root, Communicator<GpucclBackend>* comm);             \
//     template GPU_DEVICE void                                                                                        \
//     Coordinator<GpucclBackend, LaunchMode::HostDriven>::Reduce<ThreadGroup::WARP, ReductionOperator::OP, TYPE>(     \
//         TYPE * buffer, size_t count, int root, Communicator<GpucclBackend>* comm);                                  \
//     template GPU_DEVICE void                                                                                        \
//     Coordinator<GpucclBackend, LaunchMode::HostDriven>::Reduce<ThreadGroup::THREAD, ReductionOperator::OP, TYPE>(   \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, int root, Communicator<GpucclBackend>* comm);             \
//     template GPU_DEVICE void                                                                                        \
//     Coordinator<GpucclBackend, LaunchMode::HostDriven>::Reduce<ThreadGroup::THREAD, ReductionOperator::OP, TYPE>(   \
//         TYPE * buffer, size_t count, int root, Communicator<GpucclBackend>* comm);

// UNC_GPUCCL_REPT_REDUCE(DECL_FUNC)

}  // namespace uniconn
