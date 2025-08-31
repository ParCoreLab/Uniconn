#include "common.hpp"
#include "uniconn/gpushmem/communicator.hpp"
namespace uniconn {

GPU_HOST Communicator<GpushmemBackend>::Communicator() {
    nvshmem_team_config_t config;
    nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, 0, 1, nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD), &config, 0,
                               &nvshmem_comm);
    GPU_RT_CALL(UncGpuMalloc(&comm_d, sizeof(Communicator<GpushmemBackend>)));
    GPU_RT_CALL(UncGpuMemcpy(comm_d, this, sizeof(Communicator<GpushmemBackend>), UncGpuMemcpyHostToDevice));
}

// GPU_HOST Communicator<GpushmemBackend>* Communicator<GpushmemBackend>::toDevice() { return comm_d; }

// GPU_UNIFIED int Communicator<GpushmemBackend>::GlobalSize() { return nvshmem_team_n_pes(nvshmem_comm); }

// GPU_UNIFIED int Communicator<GpushmemBackend>::GlobalRank() { return nvshmem_team_my_pe(nvshmem_comm); }

// template <ThreadGroup SCOPE>
// GPU_DEVICE void Communicator<GpushmemBackend>::Barrier() {
//     if constexpr (SCOPE == ThreadGroup::BLOCK) {
//         nvshmemx_barrier_block(nvshmem_comm);
//     } else if constexpr (SCOPE == ThreadGroup::WARP) {
//         nvshmemx_barrier_warp(nvshmem_comm);
//     } else if constexpr (SCOPE == ThreadGroup::THREAD) {
//         nvshmem_barrier(nvshmem_comm);
//     }
// }

// GPU_HOST void Communicator<GpushmemBackend>::Barrier(UncGpuStream_t stream) {
//     nvshmemx_barrier_on_stream(nvshmem_comm, stream);
// }

GPU_HOST Communicator<GpushmemBackend>::~Communicator() {
    GPU_RT_CALL(UncGpuFree(comm_d));
    nvshmem_team_destroy(nvshmem_comm);
}

// template GPU_DEVICE void Communicator<GpushmemBackend>::Barrier<ThreadGroup::BLOCK>();
// template GPU_DEVICE void Communicator<GpushmemBackend>::Barrier<ThreadGroup::WARP>();
// template GPU_DEVICE void Communicator<GpushmemBackend>::Barrier<ThreadGroup::THREAD>();

}  // namespace uniconn
