#ifndef __UNICONN_INCLUDE_UNICONN_GPUSHMEM_COMMUNICATOR_HPP_
#define __UNICONN_INCLUDE_UNICONN_GPUSHMEM_COMMUNICATOR_HPP_

#include "common.hpp"
#include "uniconn/interfaces/communicator.hpp"
namespace uniconn {
template <>
struct Communicator<GpushmemBackend> {
    nvshmem_team_t nvshmem_comm;
    Communicator<GpushmemBackend>* comm_d;

    GPU_HOST Communicator();
    GPU_HOST inline Communicator<GpushmemBackend>* toDevice() { return comm_d; }
    GPU_UNIFIED inline int GlobalSize() { return nvshmem_team_n_pes(nvshmem_comm); }
    GPU_UNIFIED inline int GlobalRank() { return nvshmem_team_my_pe(nvshmem_comm); }
    template <ThreadGroup SCOPE>
    GPU_DEVICE inline void Barrier() {
        if constexpr (SCOPE == ThreadGroup::BLOCK) {
            nvshmemx_barrier_block(nvshmem_comm);
        } else if constexpr (SCOPE == ThreadGroup::WARP) {
            nvshmemx_barrier_warp(nvshmem_comm);
        } else if constexpr (SCOPE == ThreadGroup::THREAD) {
            nvshmem_barrier(nvshmem_comm);
        }
    }
    GPU_HOST inline void Barrier(UncGpuStream_t stream){nvshmemx_barrier_on_stream(nvshmem_comm, stream);}
    GPU_HOST ~Communicator();
};

}  // namespace uniconn
#endif  // __UNICONN_INCLUDE_UNICONN_GPUSHMEM_COMMUNICATOR_HPP_
