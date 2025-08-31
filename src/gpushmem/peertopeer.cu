#include "common.hpp"
#include "uniconn/gpushmem/coordinator.hpp"
// namespace cg = cooperative_groups;
namespace uniconn {
namespace internal {
namespace gpushmem {
GPU_KERNEL void signal_device(uint64_t* sig_addr, uint64_t signal_val, int dest_process_id,
                              Communicator<GpushmemBackend>* comm) {
    nvshmem_fence();
    nvshmemx_signal_op(sig_addr, 1, NVSHMEM_SIGNAL_ADD,
                       nvshmem_team_translate_pe(comm->nvshmem_comm, dest_process_id, NVSHMEM_TEAM_WORLD));
}
}  // namespace gpushmem
}  // namespace internal

// template <LaunchMode MODE>
// template <typename TYPE>
// GPU_HOST void Coordinator<GpushmemBackend, MODE>::Acknowledge(TYPE* dest_buffer, size_t buffer_size,
//                                                               uint64_t* signal_location, uint64_t signal_val,
//                                                               int src_process_id, Communicator<GpushmemBackend>*
//                                                               comm) {
//     if constexpr (MODE == LaunchMode::HostDriven || MODE == LaunchMode::LimitedDevice) {
//         nvshmemx_signal_wait_until_on_stream(signal_location, NVSHMEM_CMP_GE, signal_val, this->stream);
//     }
// }

// template <LaunchMode MODE>
// template <ThreadGroup SCOPE, typename TYPE>
// GPU_DEVICE void Coordinator<GpushmemBackend, MODE>::Acknowledge(TYPE* dest_buffer, size_t buffer_size,
//                                                                 uint64_t* signal_location, uint64_t signal_val,
//                                                                 int src_process_id,
//                                                                 Communicator<GpushmemBackend>* comm) {
//     if constexpr (MODE == LaunchMode::FullDevice) {
//         if constexpr (SCOPE == ThreadGroup::BLOCK) {
//             cg::thread_block cta = cg::this_thread_block();
//             if (cta.thread_rank() == 0) {
//                 nvshmem_signal_wait_until(signal_location, NVSHMEM_CMP_GE, signal_val);
//             }
//             cta.sync();
//         } else if constexpr (SCOPE == ThreadGroup::WARP) {
//             cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
//             if (warp.thread_rank() == 0) {
//                 nvshmem_signal_wait_until(signal_location, NVSHMEM_CMP_GE, signal_val);
//             }
//             warp.sync();
//         } else if constexpr (SCOPE == ThreadGroup::THREAD) {
//             nvshmem_signal_wait_until(signal_location, NVSHMEM_CMP_GE, signal_val);
//         }
//     }
// }

// #define DECL_FUNC(TYPENAME, TYPE, MODE)                                                                             \
//     template <>                                                                                                     \
//     template <>                                                                                                     \
//     GPU_HOST void Coordinator<GpushmemBackend, MODE>::Post<TYPE>(                                                   \
//         TYPE * src_buffer, TYPE * dest_buffer, size_t buffer_size, uint64_t* signal_location, uint64_t signal_val,  \
//         int dest_process_id, Communicator<GpushmemBackend>* comm) {                                                 \
//         if constexpr (MODE == LaunchMode::HostDriven) {                                                             \
//             nvshmemx_##TYPENAME##_put_signal_nbi_on_stream(                                                         \
//                 dest_buffer, src_buffer, buffer_size, signal_location, 1, NVSHMEM_SIGNAL_ADD,              \
//                 nvshmem_team_translate_pe(comm->nvshmem_comm, dest_process_id, NVSHMEM_TEAM_WORLD), stream);        \
//                                                                                                                     \
//         } else if constexpr (MODE == LaunchMode::LimitedDevice) {                                                   \
//             signal_device<<<1, 1, 0, stream>>>(signal_location, signal_val, dest_process_id, comm->toDevice());     \
//         }                                                                                                           \
//     }                                                                                                               \
//                                                                                                                     \
//     template GPU_HOST void Coordinator<GpushmemBackend, MODE>::Acknowledge<TYPE>(                                   \
//         TYPE * dest_buffer, size_t buffer_size, uint64_t* signal_location, uint64_t signal_val, int src_process_id, \
//         Communicator<GpushmemBackend>* comm);

// UNC_SHMEM_REPT_STANDARD_TYPE_MODE_HOST(DECL_FUNC)
// #undef DECL_FUNC

// #define DECL_FUNC(TYPENAME, TYPE, MODE, SCOPE)                                                                      \
//                                                                                                                     \
//     template <>                                                                                                     \
//     template <>                                                                                                     \
//     GPU_DEVICE void Coordinator<GpushmemBackend, MODE>::Post<SCOPE, TYPE>(                                          \
//         TYPE * src_buffer, TYPE * dest_buffer, size_t buffer_size, uint64_t* signal_location, uint64_t signal_val,  \
//         int dest_process_id, Communicator<GpushmemBackend>* comm) {                                                 \
//         if constexpr (MODE == LaunchMode::LimitedDevice) {                                                          \
//             if constexpr (SCOPE == ThreadGroup::BLOCK) {                                                            \
//                 nvshmemx_##TYPENAME##_put_nbi_block(                                                                \
//                     dest_buffer, src_buffer, buffer_size,                                                           \
//                     nvshmem_team_translate_pe(comm->nvshmem_comm, dest_process_id, NVSHMEM_TEAM_WORLD));            \
//             } else if constexpr (SCOPE == ThreadGroup::WARP) {                                                      \
//                 nvshmemx_##TYPENAME##_put_nbi_warp(                                                                 \
//                     dest_buffer, src_buffer, buffer_size,                                                           \
//                     nvshmem_team_translate_pe(comm->nvshmem_comm, dest_process_id, NVSHMEM_TEAM_WORLD));            \
//             } else if constexpr (SCOPE == ThreadGroup::THREAD) {                                                    \
//                 nvshmem_##TYPENAME##_put_nbi(                                                                       \
//                     dest_buffer, src_buffer, buffer_size,                                                           \
//                     nvshmem_team_translate_pe(comm->nvshmem_comm, dest_process_id, NVSHMEM_TEAM_WORLD));            \
//             }                                                                                                       \
//         } else if constexpr (MODE == LaunchMode::FullDevice) {                                                      \
//             if constexpr (SCOPE == ThreadGroup::BLOCK) {                                                            \
//                 nvshmemx_##TYPENAME##_put_signal_nbi_block(                                                         \
//                     dest_buffer, src_buffer, buffer_size, signal_location, 1, NVSHMEM_SIGNAL_ADD,          \
//                     nvshmem_team_translate_pe(comm->nvshmem_comm, dest_process_id, NVSHMEM_TEAM_WORLD));            \
//             } else if constexpr (SCOPE == ThreadGroup::WARP) {                                                      \
//                 nvshmemx_##TYPENAME##_put_signal_nbi_warp(                                                          \
//                     dest_buffer, src_buffer, buffer_size, signal_location, 1, NVSHMEM_SIGNAL_ADD,          \
//                     nvshmem_team_translate_pe(comm->nvshmem_comm, dest_process_id, NVSHMEM_TEAM_WORLD));            \
//             } else if constexpr (SCOPE == ThreadGroup::THREAD) {                                                    \
//                 nvshmem_##TYPENAME##_put_signal_nbi(                                                                \
//                     dest_buffer, src_buffer, buffer_size, signal_location, 1, NVSHMEM_SIGNAL_ADD,          \
//                     nvshmem_team_translate_pe(comm->nvshmem_comm, dest_process_id, NVSHMEM_TEAM_WORLD));            \
//             }                                                                                                       \
//         }                                                                                                           \
//     }                                                                                                               \
//                                                                                                                     \
//     template GPU_DEVICE void Coordinator<GpushmemBackend, MODE>::Acknowledge<SCOPE, TYPE>(                          \
//         TYPE * dest_buffer, size_t buffer_size, uint64_t* signal_location, uint64_t signal_val, int src_process_id, \
//         Communicator<GpushmemBackend>* comm);

// UNC_SHMEM_REPT_STANDARD_TYPE_MODE_GROUP_DEVICE(DECL_FUNC)
// #undef DECL_FUNC

}  // namespace uniconn
