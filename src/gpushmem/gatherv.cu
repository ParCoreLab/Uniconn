#include "uniconn/gpushmem/coordinator.hpp"
#include "common.hpp"
namespace uniconn {
// #define DECL_FUNC(TYPENAME, TYPE, MODE)                                                                              \
//     template <>                                                                                                      \
//     template <>                                                                                                      \
//     GPU_HOST void Coordinator<GpushmemBackend, MODE>::Gatherv<TYPE>(const TYPE* sendbuf, TYPE* recvbuf,              \
//                                                                     size_t* counts, size_t* displs, int root,        \
//                                                                     Communicator<GpushmemBackend>* comm) {           \
//         if constexpr (MODE == LaunchMode::HostDriven || MODE == LaunchMode::LimitedDevice) {                         \
//             comm->Barrier(stream);                                                                                   \
//             nvshmemx_##TYPENAME##_put_nbi_on_stream(                                                                 \
//                 recvbuf + displs[comm->GlobalRank()], sendbuf, counts[comm->GlobalRank()],                           \
//                 nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD), stream);                    \
//             comm->Barrier(stream);                                                                                   \
//         }                                                                                                            \
//     }                                                                                                                \
//                                                                                                                      \
//     template <>                                                                                                      \
//     template <>                                                                                                      \
//     GPU_HOST void Coordinator<GpushmemBackend, MODE>::Gatherv<TYPE>(TYPE * buffer, size_t* counts, size_t* displs,   \
//                                                                     int root, Communicator<GpushmemBackend>* comm) { \
//         Gatherv((buffer + displs[comm->GlobalRank()]), buffer, counts, displs, root, comm);                          \
//     }

// UNC_SHMEM_REPT_STANDARD_TYPE_MODE_HOST(DECL_FUNC)
// #undef DECL_FUNC

// #define DECL_FUNC(TYPENAME, TYPE, MODE, SCOPE)                                                                         \
//     template <>                                                                                                        \
//     template <>                                                                                                        \
//     GPU_DEVICE void Coordinator<GpushmemBackend, MODE>::Gatherv<SCOPE, TYPE>(const TYPE* sendbuf, TYPE* recvbuf,       \
//                                                                              size_t* counts, size_t* displs, int root, \
//                                                                              Communicator<GpushmemBackend>* comm) {    \
//         if constexpr (MODE == LaunchMode::FullDevice) {                                                                \
//             comm->Barrier<SCOPE>();                                                                                    \
//             if constexpr (SCOPE == ThreadGroup::BLOCK) {                                                               \
//                 nvshmemx_##TYPENAME##_put_nbi_block(                                                                   \
//                     recvbuf + displs[comm->GlobalRank()], sendbuf, counts[comm->GlobalRank()],                         \
//                     nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD));                          \
//             } else if constexpr (SCOPE == ThreadGroup::WARP) {                                                         \
//                 nvshmemx_##TYPENAME##_put_nbi_warp(                                                                    \
//                     recvbuf + displs[comm->GlobalRank()], sendbuf, counts[comm->GlobalRank()],                         \
//                     nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD));                          \
//             } else if constexpr (SCOPE == ThreadGroup::THREAD) {                                                       \
//                 nvshmem_##TYPENAME##_put_nbi(recvbuf + displs[comm->GlobalRank()], sendbuf,                            \
//                                              counts[comm->GlobalRank()],                                               \
//                                              nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD)); \
//             }                                                                                                          \
//             comm->Barrier<SCOPE>();                                                                                    \
//         }                                                                                                              \
//     }                                                                                                                  \
//                                                                                                                        \
//     template <>                                                                                                        \
//     template <>                                                                                                        \
//     GPU_DEVICE void Coordinator<GpushmemBackend, MODE>::Gatherv<SCOPE, TYPE>(                                          \
//         TYPE * buffer, size_t* counts, size_t* displs, int root, Communicator<GpushmemBackend>* comm) {                \
//         Gatherv<SCOPE>((buffer + displs[comm->GlobalRank()]), buffer, counts, displs, root, comm);                     \
//     }

// UNC_SHMEM_REPT_STANDARD_TYPE_MODE_GROUP_DEVICE(DECL_FUNC)
// #undef DECL_FUNC

}  // namespace uniconn
