#include "uniconn/gpushmem/coordinator.hpp"
#include "common.hpp"
namespace uniconn {

// #define DECL_FUNC(TYPENAME, TYPE, MODE)                                                                                \
//     template <>                                                                                                        \
//     template <>                                                                                                        \
//     GPU_HOST void Coordinator<GpushmemBackend, MODE>::AlltoAll<TYPE>(const TYPE* sendbuf, TYPE* recvbuf, size_t count, \
//                                                                      Communicator<GpushmemBackend>* comm) {            \
//         if constexpr (MODE == LaunchMode::HostDriven || MODE == LaunchMode::LimitedDevice) {                           \
//             nvshmemx_##TYPENAME##_alltoall_on_stream(comm->nvshmem_comm, recvbuf, sendbuf, count, stream);             \
//         }                                                                                                              \
//     }

// UNC_SHMEM_REPT_STANDARD_TYPE_MODE_HOST(DECL_FUNC)
// #undef DECL_FUNC

// #define DECL_FUNC(TYPENAME, TYPE, MODE, SCOPE)                                                     \
//     template <>                                                                                    \
//     template <>                                                                                    \
//     GPU_DEVICE void Coordinator<GpushmemBackend, MODE>::AlltoAll<SCOPE, TYPE>(                     \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, Communicator<GpushmemBackend>* comm) {   \
//         if constexpr (MODE == LaunchMode::FullDevice) {                                            \
//             if constexpr (SCOPE == ThreadGroup::BLOCK) {                                           \
//                 nvshmemx_##TYPENAME##_alltoall_block(comm->nvshmem_comm, recvbuf, sendbuf, count); \
//             } else if constexpr (SCOPE == ThreadGroup::WARP) {                                     \
//                 nvshmemx_##TYPENAME##_alltoall_warp(comm->nvshmem_comm, recvbuf, sendbuf, count);  \
//             } else if constexpr (SCOPE == ThreadGroup::THREAD) {                                   \
//                 nvshmem_##TYPENAME##_alltoall(comm->nvshmem_comm, recvbuf, sendbuf, count);        \
//             }                                                                                      \
//         }                                                                                          \
//     }

// UNC_SHMEM_REPT_STANDARD_TYPE_MODE_GROUP_DEVICE(DECL_FUNC)
// #undef DECL_FUNC

}  // namespace uniconn
