#include "uniconn/gpushmem/coordinator.hpp"
#include "common.hpp"
namespace uniconn {
// #define DECL_FUNC(TYPENAME, TYPE, MODE)                                                                      \
//     template <>                                                                                              \
//     template <>                                                                                              \
//     GPU_HOST void Coordinator<GpushmemBackend, MODE>::Broadcast<TYPE>(TYPE * buffer, size_t count, int root, \
//                                                                       Communicator<GpushmemBackend>* comm) { \
//         if constexpr (MODE == LaunchMode::HostDriven || MODE == LaunchMode::LimitedDevice) {                 \
//             nvshmemx_##TYPENAME##_broadcast_on_stream(                                                       \
//                 comm->nvshmem_comm, buffer, buffer, count,                                                   \
//                 nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD), stream);            \
//         }                                                                                                    \
//     }

// UNC_SHMEM_REPT_STANDARD_TYPE_MODE_HOST(DECL_FUNC)
// #undef DECL_FUNC

// #define DECL_FUNC(TYPENAME, TYPE, MODE, SCOPE)                                                                        \
//     template <>                                                                                                       \
//     template <>                                                                                                       \
//     GPU_DEVICE void Coordinator<GpushmemBackend, MODE>::Broadcast<SCOPE, TYPE>(TYPE * buffer, size_t count, int root, \
//                                                                                Communicator<GpushmemBackend>* comm) { \
//         if constexpr (MODE == LaunchMode::FullDevice) {                                                               \
//             if constexpr (SCOPE == ThreadGroup::BLOCK) {                                                              \
//                 nvshmemx_##TYPENAME##_broadcast_block(                                                                \
//                     comm->nvshmem_comm, buffer, buffer, count,                                                        \
//                     nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD));                         \
//             } else if constexpr (SCOPE == ThreadGroup::WARP) {                                                        \
//                 nvshmemx_##TYPENAME##_broadcast_warp(                                                                 \
//                     comm->nvshmem_comm, buffer, buffer, count,                                                        \
//                     nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD));                         \
//             } else if constexpr (SCOPE == ThreadGroup::THREAD) {                                                      \
//                 nvshmem_##TYPENAME##_broadcast(                                                                       \
//                     comm->nvshmem_comm, buffer, buffer, count,                                                        \
//                     nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD));                         \
//             }                                                                                                         \
//         }                                                                                                             \
//     }

// UNC_SHMEM_REPT_STANDARD_TYPE_MODE_GROUP_DEVICE(DECL_FUNC)
// #undef DECL_FUNC

}  // namespace uniconn
