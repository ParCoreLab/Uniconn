#include "common.hpp"
#include "uniconn/gpumpi/coordinator.hpp"
namespace uniconn {
// template <ReductionOperator OP, typename T>
// GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::Reduce(const T* sendbuf, T* recvbuf, size_t count,
//                                                                       int root, Communicator<MPIBackend>* comm) {
//     if (sendbuf == internal::IN_PLACE<T> && comm->GlobalRank() != root) {
//         sendbuf = recvbuf;
//     }
//     if (is_grouped == 0) {
//         if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
//             GPU_RT_CALL(UncGpuStreamSynchronize(stream));
//         }
//         MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Reduce)(internal::mpi::buf_or_inplace(sendbuf), recvbuf, count,
//                                                       internal::mpi::TypeMap<T>(), internal::mpi::ReductOp2MPI_Op<OP>(),
//                                                       root, comm->mpi_comm));
//     } else {
//         MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Ireduce)(
//             internal::mpi::buf_or_inplace(sendbuf), recvbuf, count, internal::mpi::TypeMap<T>(),
//             internal::mpi::ReductOp2MPI_Op<OP>(), root, comm->mpi_comm, &internal_reqs.emplace_back(MPI_REQUEST_NULL)));
//     }
// }
// template <ReductionOperator OP, typename T>
// GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::Reduce(T* buffer, size_t count, int root,
//                                                                       Communicator<MPIBackend>* comm) {
//     Reduce<OP>(internal::IN_PLACE<T>, buffer, count, root, comm);
// }

// template <ThreadGroup SCOPE, ReductionOperator OP, typename T>
// GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Reduce(const T* sendbuf, T* recvbuf, size_t count,
//                                                                         int root, Communicator<MPIBackend>* comm) {}
// template <ThreadGroup SCOPE, ReductionOperator OP, typename T>
// GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Reduce(T* buffer, size_t count, int root,
//                                                                         Communicator<MPIBackend>* comm) {}

// #define DECL_FUNC(TYPE, OP)                                                                                      \
//     template GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::Reduce<ReductionOperator::OP, TYPE>( \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, int root, Communicator<MPIBackend>* comm);             \
//     template GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::Reduce<ReductionOperator::OP, TYPE>( \
//         TYPE * buffer, size_t count, int root, Communicator<MPIBackend>* comm);                                  \
//     template GPU_DEVICE void                                                                                     \
//     Coordinator<MPIBackend, LaunchMode::HostDriven>::Reduce<ThreadGroup::BLOCK, ReductionOperator::OP, TYPE>(    \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, int root, Communicator<MPIBackend>* comm);             \
//     template GPU_DEVICE void                                                                                     \
//     Coordinator<MPIBackend, LaunchMode::HostDriven>::Reduce<ThreadGroup::BLOCK, ReductionOperator::OP, TYPE>(    \
//         TYPE * buffer, size_t count, int root, Communicator<MPIBackend>* comm);                                  \
//     template GPU_DEVICE void                                                                                     \
//     Coordinator<MPIBackend, LaunchMode::HostDriven>::Reduce<ThreadGroup::WARP, ReductionOperator::OP, TYPE>(     \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, int root, Communicator<MPIBackend>* comm);             \
//     template GPU_DEVICE void                                                                                     \
//     Coordinator<MPIBackend, LaunchMode::HostDriven>::Reduce<ThreadGroup::WARP, ReductionOperator::OP, TYPE>(     \
//         TYPE * buffer, size_t count, int root, Communicator<MPIBackend>* comm);                                  \
//     template GPU_DEVICE void                                                                                     \
//     Coordinator<MPIBackend, LaunchMode::HostDriven>::Reduce<ThreadGroup::THREAD, ReductionOperator::OP, TYPE>(   \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, int root, Communicator<MPIBackend>* comm);             \
//     template GPU_DEVICE void                                                                                     \
//     Coordinator<MPIBackend, LaunchMode::HostDriven>::Reduce<ThreadGroup::THREAD, ReductionOperator::OP, TYPE>(   \
//         TYPE * buffer, size_t count, int root, Communicator<MPIBackend>* comm);

// UNC_GPUMPI_REPT_REDUCE(DECL_FUNC)

}  // namespace uniconn
