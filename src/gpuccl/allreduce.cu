#include "uniconn/gpuccl/coordinator.hpp"
#include "common.hpp"
namespace uniconn {
// template <ReductionOperator OP, typename T>
// GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllReduce(const T* sendbuf, T* recvbuf, size_t count,
//                                                                             Communicator<GpucclBackend>* comm) {
   
//     NCCL_CALL(ncclAllReduce((const void*)sendbuf, (void*)recvbuf, count, internal::gpuccl::TypeMap<T>(),
//                             internal::gpuccl::ReductOp2ncclRedOp<OP>(), comm->nccl_comm, stream));
// }
// template <ReductionOperator OP, typename T>
// GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllReduce(T* buffer, size_t count,
//                                                                             Communicator<GpucclBackend>* comm) {
//     AllReduce<OP>(buffer, buffer, count, comm);
// }

// template <ThreadGroup SCOPE, ReductionOperator OP, typename T>
// GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllReduce(const T* sendbuf, T* recvbuf,
//                                                                               size_t count,
//                                                                               Communicator<GpucclBackend>* comm) {}
// template <ThreadGroup SCOPE, ReductionOperator OP, typename T>
// GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllReduce(T* buffer, size_t count,
//                                                                               Communicator<GpucclBackend>* comm) {}

// #define DECL_FUNC(TYPE, OP)                                                                                            \
//     template GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllReduce<ReductionOperator::OP, TYPE>( \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, Communicator<GpucclBackend>* comm);                          \
//     template GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllReduce<ReductionOperator::OP, TYPE>( \
//         TYPE * buffer, size_t count, Communicator<GpucclBackend>* comm);                                               \
//     template GPU_DEVICE void                                                                                           \
//     Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllReduce<ThreadGroup::BLOCK, ReductionOperator::OP, TYPE>(    \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, Communicator<GpucclBackend>* comm);                          \
//     template GPU_DEVICE void                                                                                           \
//     Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllReduce<ThreadGroup::BLOCK, ReductionOperator::OP, TYPE>(    \
//         TYPE * buffer, size_t count, Communicator<GpucclBackend>* comm);                                               \
//     template GPU_DEVICE void                                                                                           \
//     Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllReduce<ThreadGroup::WARP, ReductionOperator::OP, TYPE>(     \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, Communicator<GpucclBackend>* comm);                          \
//     template GPU_DEVICE void                                                                                           \
//     Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllReduce<ThreadGroup::WARP, ReductionOperator::OP, TYPE>(     \
//         TYPE * buffer, size_t count, Communicator<GpucclBackend>* comm);                                               \
//     template GPU_DEVICE void                                                                                           \
//     Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllReduce<ThreadGroup::THREAD, ReductionOperator::OP, TYPE>(   \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, Communicator<GpucclBackend>* comm);                          \
//     template GPU_DEVICE void                                                                                           \
//     Coordinator<GpucclBackend, LaunchMode::HostDriven>::AllReduce<ThreadGroup::THREAD, ReductionOperator::OP, TYPE>(   \
//         TYPE * buffer, size_t count, Communicator<GpucclBackend>* comm);

// UNC_GPUCCL_REPT_REDUCE(DECL_FUNC)

}  // namespace uniconn
