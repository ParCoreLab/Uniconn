#include "uniconn/gpushmem/coordinator.hpp"
#include "common.hpp"
namespace uniconn {
// #define DECL_FUNC(TYPENAME, TYPE, MODE)                                                                               \
//     template <>                                                                                                       \
//     template <>                                                                                                       \
//     GPU_HOST void Coordinator<GpushmemBackend, MODE>::Scatter<TYPE>(const TYPE* sendbuf, TYPE* recvbuf, size_t count, \
//                                                                     int root, Communicator<GpushmemBackend>* comm) {  \
//         if constexpr (MODE == LaunchMode::HostDriven || MODE == LaunchMode::LimitedDevice) {                          \
//             comm->Barrier(stream);                                                                                    \
//             nvshmemx_##TYPENAME##_get_nbi_on_stream(                                                                  \
//                 recvbuf, sendbuf + comm->GlobalRank() * count, count,                                                 \
//                 nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD), stream);                     \
//             comm->Barrier(stream);                                                                                    \
//         }                                                                                                             \
//     }                                                                                                                 \
//                                                                                                                       \
//     template <>                                                                                                       \
//     template <>                                                                                                       \
//     GPU_HOST void Coordinator<GpushmemBackend, MODE>::Scatter<TYPE>(TYPE * buffer, size_t count, int root,            \
//                                                                     Communicator<GpushmemBackend>* comm) {            \
//         Scatter((buffer + comm->GlobalRank() * count), buffer, count, root, comm);                                    \
//     }

// UNC_SHMEM_REPT_STANDARD_TYPE_MODE_HOST(DECL_FUNC)
// #undef DECL_FUNC

// #define DECL_FUNC(TYPENAME, TYPE, MODE, SCOPE)                                                                         \
//     template <>                                                                                                        \
//     template <>                                                                                                        \
//     GPU_DEVICE void Coordinator<GpushmemBackend, MODE>::Scatter<SCOPE, TYPE>(                                          \
//         const TYPE* sendbuf, TYPE* recvbuf, size_t count, int root, Communicator<GpushmemBackend>* comm) {             \
//         if constexpr (MODE == LaunchMode::FullDevice) {                                                                \
//             comm->Barrier<SCOPE>();                                                                                    \
//             if constexpr (SCOPE == ThreadGroup::BLOCK) {                                                               \
//                 nvshmemx_##TYPENAME##_get_nbi_block(                                                                   \
//                     recvbuf, sendbuf + comm->GlobalRank() * count, count,                                              \
//                     nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD));                          \
//             } else if constexpr (SCOPE == ThreadGroup::WARP) {                                                         \
//                 nvshmemx_##TYPENAME##_get_nbi_warp(                                                                    \
//                     recvbuf, sendbuf + comm->GlobalRank() * count, count,                                              \
//                     nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD));                          \
//             } else if constexpr (SCOPE == ThreadGroup::THREAD) {                                                       \
//                 nvshmem_##TYPENAME##_get_nbi(recvbuf, sendbuf + comm->GlobalRank() * count, count,                     \
//                                              nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD)); \
//             }                                                                                                          \
//             comm->Barrier<SCOPE>();                                                                                    \
//         }                                                                                                              \
//     }                                                                                                                  \
//                                                                                                                        \
//     template <>                                                                                                        \
//     template <>                                                                                                        \
//     GPU_DEVICE void Coordinator<GpushmemBackend, MODE>::Scatter<SCOPE, TYPE>(TYPE * buffer, size_t count, int root,    \
//                                                                              Communicator<GpushmemBackend>* comm) {    \
//         Scatter<SCOPE>((buffer + comm->GlobalRank() * count), buffer, count, root, comm);                              \
//     }

// UNC_SHMEM_REPT_STANDARD_TYPE_MODE_GROUP_DEVICE(DECL_FUNC)
// #undef DECL_FUNC

}  // namespace uniconn
