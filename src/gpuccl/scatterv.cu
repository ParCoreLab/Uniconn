#include "common.hpp"
#include "uniconn/gpuccl/coordinator.hpp"
namespace uniconn {
// template <typename T>
// GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Scatterv(const T* sendbuf, T* recvbuf, size_t* counts,
//                                                                            size_t* displs, int root,
//                                                                            Communicator<GpucclBackend>* comm) {
//     NCCL_CALL(ncclGroupStart());
//     if (counts[comm->GlobalRank()] != 0) {
//         NCCL_CALL(ncclRecv((void*)recvbuf, counts[comm->GlobalRank()], internal::gpuccl::TypeMap<T>(), root,
//                            comm->nccl_comm, stream));
//     }

//     if (comm->GlobalRank() == root) {
//         for (size_t i = 0; i < comm->GlobalSize(); ++i) {
//             if (counts[i] != 0) {
//                 NCCL_CALL(ncclSend((const void*)(sendbuf + displs[i]), counts[i], internal::gpuccl::TypeMap<T>(), i,
//                                    comm->nccl_comm, stream));
//             }
//         }
//     }
//     NCCL_CALL(ncclGroupEnd());
// }

// template <typename T>
// GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Scatterv(T* buffer, size_t* counts, size_t* displs,
//                                                                            int root,
//                                                                            Communicator<GpucclBackend>* comm) {
//     Scatterv((buffer + displs[comm->GlobalRank()]), buffer, counts, displs, root, comm);
// }

// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Scatterv(const T* sendbuf, T* recvbuf,
//                                                                              size_t* counts, size_t* displs, int root,
//                                                                              Communicator<GpucclBackend>* comm) {}

// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Scatterv(T* buffer, size_t* counts, size_t* displs,
//                                                                              int root,
//                                                                              Communicator<GpucclBackend>* comm) {}

// #define DECL_FUNC(TYPE)                                                                                               \
//     template GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Scatterv<TYPE>(                        \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t* counts, size_t* displs, int root,                                 \
//         Communicator<GpucclBackend>* comm);                                                                           \
//     template GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Scatterv<TYPE>(                        \
//         TYPE * buffer, size_t* counts, size_t* displs, int root, Communicator<GpucclBackend>* comm);                  \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Scatterv<ThreadGroup::BLOCK, TYPE>(  \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t* counts, size_t* displs, int root,                                 \
//         Communicator<GpucclBackend>* comm);                                                                           \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Scatterv<ThreadGroup::BLOCK, TYPE>(  \
//         TYPE * buffer, size_t* counts, size_t* displs, int root, Communicator<GpucclBackend>* comm);                  \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Scatterv<ThreadGroup::WARP, TYPE>(   \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t* counts, size_t* displs, int root,                                 \
//         Communicator<GpucclBackend>* comm);                                                                           \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Scatterv<ThreadGroup::WARP, TYPE>(   \
//         TYPE * buffer, size_t* counts, size_t* displs, int root, Communicator<GpucclBackend>* comm);                  \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Scatterv<ThreadGroup::THREAD, TYPE>( \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t* counts, size_t* displs, int root,                                 \
//         Communicator<GpucclBackend>* comm);                                                                           \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Scatterv<ThreadGroup::THREAD, TYPE>( \
//         TYPE * buffer, size_t* counts, size_t* displs, int root, Communicator<GpucclBackend>* comm);

// UNC_GPUCCL_REPT(DECL_FUNC)

}  // namespace uniconn
