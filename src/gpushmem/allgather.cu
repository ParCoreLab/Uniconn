#include "uniconn/gpushmem/coordinator.hpp"
#include "common.hpp"
namespace uniconn {
// #define DECL_FUNC(TYPENAME, TYPE, MODE)                                                                             \
//     template <>                                                                                                     \
//     template <>                                                                                                     \
//     GPU_HOST void Coordinator<GpushmemBackend, MODE>::AllGather<TYPE>(TYPE * sendbuf, TYPE * recvbuf, size_t count, \
//                                                                       Communicator<GpushmemBackend>* comm) {        \
//         if constexpr (MODE == LaunchMode::HostDriven || MODE == LaunchMode::LimitedDevice) {                        \
//             nvshmemx_fcollectmem_on_stream(comm->nvshmem_comm, recvbuf, sendbuf, count*sizeof(T), this->stream);    \
//         }                                                                                                           \
//     }                                                                                                               \
//                                                                                                                     \
//     template <>                                                                                                     \
//     template <>                                                                                                     \
//     GPU_HOST void Coordinator<GpushmemBackend, MODE>::AllGather<TYPE>(TYPE * buffer, size_t count,                  \
//                                                                       Communicator<GpushmemBackend>* comm) {        \
//         AllGather((buffer + comm->GlobalRank() * count), buffer, count, comm);                                      \
//     }

// UNC_SHMEM_REPT_STANDARD_TYPE_MODE_HOST(DECL_FUNC)
// #undef DECL_FUNC

// #define DECL_FUNC(TYPENAME, TYPE, MODE, SCOPE)                                                                        \
//     template <>                                                                                                       \
//     template <>                                                                                                       \
//     GPU_DEVICE void Coordinator<GpushmemBackend, MODE>::AllGather<SCOPE, TYPE>(                                       \
//         TYPE * sendbuf, TYPE * recvbuf, size_t count, Communicator<GpushmemBackend>* comm) {                          \
//         if constexpr (MODE == LaunchMode::FullDevice) {                                                               \
//             if constexpr (SCOPE == ThreadGroup::BLOCK) {                                                              \
//                 nvshmemx_fcollectmem_block(comm->nvshmem_comm, recvbuf, sendbuf, count*sizeof(T));                    \
//             } else if constexpr (SCOPE == ThreadGroup::WARP) {                                                        \
//                 nvshmemx_fcollectmem_warp(comm->nvshmem_comm, recvbuf, sendbuf, count*sizeof(T));                     \
//             } else if constexpr (SCOPE == ThreadGroup::THREAD) {                                                      \
//                 nvshmem_fcollectmem(comm->nvshmem_comm, recvbuf, sendbuf, count*sizeof(T));                           \
//             }                                                                                                         \
//         }                                                                                                             \
//     }                                                                                                                 \
//                                                                                                                       \
//     template <>                                                                                                       \
//     template <>                                                                                                       \
//     GPU_DEVICE void Coordinator<GpushmemBackend, MODE>::AllGather<SCOPE, TYPE>(TYPE * buffer, size_t count,           \
//                                                                                Communicator<GpushmemBackend>* comm) { \
//         AllGather<SCOPE>((buffer + comm->GlobalRank() * count), buffer, count, comm);                                 \
//     }

// UNC_SHMEM_REPT_STANDARD_TYPE_MODE_GROUP_DEVICE(DECL_FUNC)
// #undef DECL_FUNC

}  // namespace uniconn
