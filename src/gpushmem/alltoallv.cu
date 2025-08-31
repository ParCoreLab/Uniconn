#include "common.hpp"
#include "uniconn/gpushmem/coordinator.hpp"
namespace uniconn {
// #define DECL_FUNC(TYPENAME, TYPE, MODE)                                                                    \
//     template <>                                                                                            \
//     template <>                                                                                            \
//     GPU_HOST void Coordinator<GpushmemBackend, MODE>::AlltoAllv<TYPE>(                                     \
//         const TYPE* sendbuf, size_t* send_counts, size_t* send_displs, TYPE* recvbuf, size_t* recv_counts, \
//         size_t* recv_displs, Communicator<GpushmemBackend>* comm) {                                        \
//         if constexpr (MODE == LaunchMode::HostDriven || MODE == LaunchMode::LimitedDevice) {               \
//             comm->Barrier(stream);                                                                         \
//             for (size_t i = 0; i < comm->GlobalSize(); i++) {                                              \
//                 nvshmemx_##TYPENAME##_put_nbi_on_stream(                                                   \
//                     recvbuf + recv_displs[comm->GlobalRank()], sendbuf + send_displs[comm->GlobalRank()],  \
//                     send_counts[comm->GlobalRank()],                                                       \
//                     nvshmem_team_translate_pe(comm->nvshmem_comm, i, NVSHMEM_TEAM_WORLD), stream);         \
//             }                                                                                              \
//             comm->Barrier(stream);                                                                         \
//         }                                                                                                  \
//     }

// UNC_SHMEM_REPT_STANDARD_TYPE_MODE_HOST(DECL_FUNC)
// #undef DECL_FUNC

// #define DECL_FUNC(TYPENAME, TYPE, MODE, SCOPE)                                                             \
//     template <>                                                                                            \
//     template <>                                                                                            \
//     GPU_DEVICE void Coordinator<GpushmemBackend, MODE>::AlltoAllv<SCOPE, TYPE>(                            \
//         const TYPE* sendbuf, size_t* send_counts, size_t* send_displs, TYPE* recvbuf, size_t* recv_counts, \
//         size_t* recv_displs, Communicator<GpushmemBackend>* comm) {                                        \
//         if constexpr (MODE == LaunchMode::FullDevice) {                                                    \
//             comm->Barrier<SCOPE>();                                                                        \
//             for (size_t i = 0; i < comm->GlobalSize(); i++) {                                              \
//                 if constexpr (SCOPE == ThreadGroup::BLOCK) {                                               \
//                     nvshmemx_##TYPENAME##_put_nbi_block(                                                   \
//                         recvbuf + recv_displs[comm->GlobalRank()], sendbuf + send_displs[comm->GlobalRank()], send_counts[comm->GlobalRank()],                \
//                         nvshmem_team_translate_pe(comm->nvshmem_comm, i, NVSHMEM_TEAM_WORLD));             \
//                 } else if constexpr (SCOPE == ThreadGroup::WARP) {                                         \
//                     nvshmemx_##TYPENAME##_put_nbi_warp(                                                    \
//                         recvbuf + recv_displs[comm->GlobalRank()], sendbuf + send_displs[comm->GlobalRank()], send_counts[comm->GlobalRank()],                \
//                         nvshmem_team_translate_pe(comm->nvshmem_comm, i, NVSHMEM_TEAM_WORLD));             \
//                 } else if constexpr (SCOPE == ThreadGroup::THREAD) {                                       \
//                     nvshmem_##TYPENAME##_put_nbi(                                                          \
//                         recvbuf + recv_displs[comm->GlobalRank()], sendbuf + send_displs[comm->GlobalRank()], send_counts[comm->GlobalRank()],                \
//                         nvshmem_team_translate_pe(comm->nvshmem_comm, i, NVSHMEM_TEAM_WORLD));             \
//                 }                                                                                          \
//             }                                                                                              \
//             comm->Barrier<SCOPE>();                                                                        \
//         }                                                                                                  \
//     }

// UNC_SHMEM_REPT_STANDARD_TYPE_MODE_GROUP_DEVICE(DECL_FUNC)
// #undef DECL_FUNC

}  // namespace uniconn
