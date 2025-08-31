#ifndef __UNICONN_INCLUDE_UNICONN_GPUSHMEM_COORDINATOR_HPP_
#define __UNICONN_INCLUDE_UNICONN_GPUSHMEM_COORDINATOR_HPP_
#include <cooperative_groups.h>
#include "common.hpp"
#include "communicator.hpp"
#include "uniconn/interfaces/coordinator.hpp"
namespace cg = cooperative_groups;
namespace uniconn {
namespace internal {
namespace gpushmem {
GPU_KERNEL void signal_device(uint64_t* sig_addr, uint64_t signal_val, int dest_process_id,
                              Communicator<GpushmemBackend>* comm);
}
}  // namespace internal

template <LaunchMode M>
class Coordinator<GpushmemBackend, M> {
   private:
    UncGpuStream_t stream;
    KernelInfo kernel_info;

   public:
    GPU_HOST Coordinator(UncGpuStream_t stream);

    template <LaunchMode MODE>
    GPU_HOST void bindKernel(void* kernel_func, dim3 grid_dims, dim3 block_dims, size_t shared_mem, void** kernel_args);

    GPU_HOST inline void LaunchKernel() {
        if constexpr (M == LaunchMode::HostDriven || M == LaunchMode::LimitedDevice) {
            kernel_info.launch(stream);

        } else if constexpr (M == LaunchMode::FullDevice) {
            GPU_RT_CALL((UncGpuError_t)nvshmemx_collective_launch(kernel_info.kernel_func, kernel_info.grid_dims,
                                                                  kernel_info.block_dims, kernel_info.kernel_args,
                                                                  kernel_info.shared_mem, stream));
        }
    }

    GPU_HOST inline void CommStart() {}

    GPU_HOST inline void CommEnd() {}

    template <typename T>
    GPU_HOST inline void Post(T* src_buffer, T* dest_buffer, size_t buffer_size, uint64_t* signal_location,
                              uint64_t signal_val, int dest_process_id, Communicator<GpushmemBackend>* comm) {
        if constexpr (M == LaunchMode::HostDriven) {
            nvshmemx_putmem_signal_nbi_on_stream(
                dest_buffer, src_buffer, buffer_size * sizeof(T), signal_location, 1, NVSHMEM_SIGNAL_ADD,
                nvshmem_team_translate_pe(comm->nvshmem_comm, dest_process_id, NVSHMEM_TEAM_WORLD), stream);

        } else if constexpr (M == LaunchMode::LimitedDevice) {
            internal::gpushmem::signal_device<<<1, 1, 0, stream>>>(signal_location, signal_val, dest_process_id,
                                                                   comm->toDevice());
        }
    }

    template <typename T>
    GPU_HOST inline void Acknowledge(T* dest_buffer, size_t buffer_size, uint64_t* signal_location, uint64_t signal_val,
                                     int src_process_id, Communicator<GpushmemBackend>* comm) {
        if constexpr (M == LaunchMode::HostDriven || M == LaunchMode::LimitedDevice) {
            nvshmemx_signal_wait_until_on_stream(signal_location, NVSHMEM_CMP_GE, signal_val, this->stream);
        }
    }

    template <typename T>
    GPU_HOST inline void AllGather(T* sendbuf, T* recvbuf, size_t count, Communicator<GpushmemBackend>* comm) {
        if constexpr (M == LaunchMode::HostDriven || M == LaunchMode::LimitedDevice) {
            nvshmemx_fcollectmem_on_stream(comm->nvshmem_comm, recvbuf, sendbuf, count * sizeof(T), this->stream);
        }
    }
    template <typename T>
    GPU_HOST inline void AllGather(T* buffer, size_t count, Communicator<GpushmemBackend>* comm) {
        AllGather((buffer + comm->GlobalRank() * count), buffer, count, comm);
    }

    template <typename T>
    GPU_HOST inline void AllGatherv(T* sendbuf, T* recvbuf, size_t* counts, size_t* displs,
                                    Communicator<GpushmemBackend>* comm) {
        if constexpr (M == LaunchMode::HostDriven || M == LaunchMode::LimitedDevice) {
            comm->Barrier(stream);
            for (size_t i = 0; i < comm->GlobalSize(); i++) {
                nvshmemx_putmem_nbi_on_stream(
                    recvbuf + displs[comm->GlobalRank()], sendbuf, counts[comm->GlobalRank()] * sizeof(T),
                    nvshmem_team_translate_pe(comm->nvshmem_comm, i, NVSHMEM_TEAM_WORLD), stream);
            }
            comm->Barrier(stream);
        }
    }
    template <typename T>
    GPU_HOST inline void AllGatherv(T* buffer, size_t* counts, size_t* displs, Communicator<GpushmemBackend>* comm) {
        AllGatherv((buffer + displs[comm->GlobalRank()]), buffer, counts, displs, comm);
    }

    template <ReductionOperator OP, typename T>
    GPU_HOST void AllReduce(const T* sendbuf, T* recvbuf, size_t count, Communicator<GpushmemBackend>* comm);
    template <ReductionOperator OP, typename T>
    GPU_HOST void AllReduce(T* buffer, size_t count, Communicator<GpushmemBackend>* comm);
    template <ReductionOperator OP, typename T>
    GPU_HOST void Reduce(const T* sendbuf, T* recvbuf, size_t count, int root, Communicator<GpushmemBackend>* comm);
    template <ReductionOperator OP, typename T>
    GPU_HOST void Reduce(T* buffer, size_t count, int root, Communicator<GpushmemBackend>* comm);

    template <typename T>
    GPU_HOST inline void AlltoAll(const T* sendbuf, T* recvbuf, size_t count, Communicator<GpushmemBackend>* comm) {
        if constexpr (M == LaunchMode::HostDriven || M == LaunchMode::LimitedDevice) {
            nvshmemx_alltoallmem_on_stream(comm->nvshmem_comm, recvbuf, sendbuf, count * sizeof(T), stream);
        }
    }
    template <typename T>
    GPU_HOST inline void AlltoAllv(const T* sendbuf, size_t* send_counts, size_t* send_displs, T* recvbuf,
                                   size_t* recv_counts, size_t* recv_displs, Communicator<GpushmemBackend>* comm) {
        if constexpr (M == LaunchMode::HostDriven || M == LaunchMode::LimitedDevice) {
            comm->Barrier(stream);
            for (size_t i = 0; i < comm->GlobalSize(); i++) {
                nvshmemx_putmem_nbi_on_stream(
                    recvbuf + recv_displs[comm->GlobalRank()], sendbuf + send_displs[comm->GlobalRank()],
                    send_counts[comm->GlobalRank()] * sizeof(T),
                    nvshmem_team_translate_pe(comm->nvshmem_comm, i, NVSHMEM_TEAM_WORLD), stream);
            }
            comm->Barrier(stream);
        }
    }

    template <typename T>
    GPU_HOST inline void Broadcast(T* buffer, size_t count, int root, Communicator<GpushmemBackend>* comm) {
        if constexpr (M == LaunchMode::HostDriven || M == LaunchMode::LimitedDevice) {
            nvshmemx_broadcastmem_on_stream(comm->nvshmem_comm, buffer, buffer, count * sizeof(T),
                                            nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD),
                                            stream);
        }
    }

    template <typename T>
    GPU_HOST inline void Gather(const T* sendbuf, T* recvbuf, size_t count, int root,
                                Communicator<GpushmemBackend>* comm) {
        if constexpr (M == LaunchMode::HostDriven || M == LaunchMode::LimitedDevice) {
            comm->Barrier(stream);
            nvshmemx_putmem_nbi_on_stream(recvbuf + (comm->GlobalRank() * count), sendbuf, count * sizeof(T),
                                          nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD),
                                          stream);
            comm->Barrier(stream);
        }
    }
    template <typename T>
    GPU_HOST inline void Gather(T* buffer, size_t count, int root, Communicator<GpushmemBackend>* comm) {
        Gather((buffer + comm->GlobalRank() * count), buffer, count, root, comm);
    }

    template <typename T>
    GPU_HOST inline void Gatherv(const T* sendbuf, T* recvbuf, size_t* counts, size_t* displs, int root,
                                 Communicator<GpushmemBackend>* comm) {
        if constexpr (M == LaunchMode::HostDriven || M == LaunchMode::LimitedDevice) {
            comm->Barrier(stream);
            nvshmemx_putmem_nbi_on_stream(
                recvbuf + displs[comm->GlobalRank()], sendbuf, counts[comm->GlobalRank()] * sizeof(T),
                nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD), stream);
            comm->Barrier(stream);
        }
    }
    template <typename T>
    GPU_HOST inline void Gatherv(T* buffer, size_t* counts, size_t* displs, int root,
                                 Communicator<GpushmemBackend>* comm) {
        Gatherv((buffer + displs[comm->GlobalRank()]), buffer, counts, displs, root, comm);
    }

    template <typename T>
    GPU_HOST inline void Scatter(const T* sendbuf, T* recvbuf, size_t count, int root,
                                 Communicator<GpushmemBackend>* comm) {
        if constexpr (M == LaunchMode::HostDriven || M == LaunchMode::LimitedDevice) {
            comm->Barrier(stream);
            nvshmemx_getmem_nbi_on_stream(recvbuf, sendbuf + (comm->GlobalRank() * count), count * sizeof(T),
                                          nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD),
                                          stream);
            comm->Barrier(stream);
        }
    }
    template <typename T>
    GPU_HOST inline void Scatter(T* buffer, size_t count, int root, Communicator<GpushmemBackend>* comm) {
        Scatter((buffer + comm->GlobalRank() * count), buffer, count, root, comm);
    }

    template <typename T>
    GPU_HOST inline void Scatterv(const T* sendbuf, T* recvbuf, size_t* counts, size_t* displs, int root,
                                  Communicator<GpushmemBackend>* comm) {
        if constexpr (M == LaunchMode::HostDriven || M == LaunchMode::LimitedDevice) {
            comm->Barrier(stream);
            nvshmemx_getmem_nbi_on_stream(
                recvbuf, sendbuf + displs[comm->GlobalRank()], counts[comm->GlobalRank()] * sizeof(T),
                nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD), stream);
            comm->Barrier(stream);
        }
    }

    template <typename T>
    GPU_HOST inline void Scatterv(T* buffer, size_t* counts, size_t* displs, int root,
                                  Communicator<GpushmemBackend>* comm) {
        Scatterv(buffer + displs[comm->GlobalRank()], buffer, counts, displs, root, comm);
    }

    GPU_HOST inline void WaitComm();

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Post(T* src_buffer, T* dest_buffer, size_t buffer_size, uint64_t* signal_location,
                                       uint64_t signal_val, int dest_process_id, Communicator<GpushmemBackend>* comm) {
        if constexpr (M == LaunchMode::LimitedDevice) {
            if constexpr (SCOPE == ThreadGroup::BLOCK) {
                nvshmemx_putmem_nbi_block(
                    dest_buffer, src_buffer, buffer_size * sizeof(T),
                    nvshmem_team_translate_pe(comm->nvshmem_comm, dest_process_id, NVSHMEM_TEAM_WORLD));
            } else if constexpr (SCOPE == ThreadGroup::WARP) {
                nvshmemx_putmem_nbi_warp(
                    dest_buffer, src_buffer, buffer_size * sizeof(T),
                    nvshmem_team_translate_pe(comm->nvshmem_comm, dest_process_id, NVSHMEM_TEAM_WORLD));
            } else if constexpr (SCOPE == ThreadGroup::THREAD) {
                nvshmem_putmem_nbi(dest_buffer, src_buffer, buffer_size * sizeof(T),
                                   nvshmem_team_translate_pe(comm->nvshmem_comm, dest_process_id, NVSHMEM_TEAM_WORLD));
            }
        } else if constexpr (M == LaunchMode::FullDevice) {
            if constexpr (SCOPE == ThreadGroup::BLOCK) {
                nvshmemx_putmem_signal_nbi_block(
                    dest_buffer, src_buffer, buffer_size * sizeof(T), signal_location, 1, NVSHMEM_SIGNAL_ADD,
                    nvshmem_team_translate_pe(comm->nvshmem_comm, dest_process_id, NVSHMEM_TEAM_WORLD));
            } else if constexpr (SCOPE == ThreadGroup::WARP) {
                nvshmemx_putmem_signal_nbi_warp(
                    dest_buffer, src_buffer, buffer_size * sizeof(T), signal_location, 1, NVSHMEM_SIGNAL_ADD,
                    nvshmem_team_translate_pe(comm->nvshmem_comm, dest_process_id, NVSHMEM_TEAM_WORLD));
            } else if constexpr (SCOPE == ThreadGroup::THREAD) {
                nvshmem_putmem_signal_nbi(
                    dest_buffer, src_buffer, buffer_size * sizeof(T), signal_location, 1, NVSHMEM_SIGNAL_ADD,
                    nvshmem_team_translate_pe(comm->nvshmem_comm, dest_process_id, NVSHMEM_TEAM_WORLD));
            }
        }
    }

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Acknowledge(T* dest_buffer, size_t buffer_size, uint64_t* signal_location,
                                              uint64_t signal_val, int src_process_id,
                                              Communicator<GpushmemBackend>* comm) {
        if constexpr (M == LaunchMode::FullDevice) {
            if constexpr (SCOPE == ThreadGroup::BLOCK) {
                cg::thread_block cta = cg::this_thread_block();
                if (cta.thread_rank() == 0) {
                    nvshmem_signal_wait_until(signal_location, NVSHMEM_CMP_GE, signal_val);
                }
                cta.sync();
            } else if constexpr (SCOPE == ThreadGroup::WARP) {
                cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
                if (warp.thread_rank() == 0) {
                    nvshmem_signal_wait_until(signal_location, NVSHMEM_CMP_GE, signal_val);
                }
                warp.sync();
            } else if constexpr (SCOPE == ThreadGroup::THREAD) {
                nvshmem_signal_wait_until(signal_location, NVSHMEM_CMP_GE, signal_val);
            }
        }
    }

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void AllGather(T* sendbuf, T* recvbuf, size_t count, Communicator<GpushmemBackend>* comm) {
        if constexpr (M == LaunchMode::FullDevice) {
            if constexpr (SCOPE == ThreadGroup::BLOCK) {
                nvshmemx_fcollectmem_block(comm->nvshmem_comm, recvbuf, sendbuf, count * sizeof(T));
            } else if constexpr (SCOPE == ThreadGroup::WARP) {
                nvshmemx_fcollectmem_warp(comm->nvshmem_comm, recvbuf, sendbuf, count * sizeof(T));
            } else if constexpr (SCOPE == ThreadGroup::THREAD) {
                nvshmem_fcollectmem(comm->nvshmem_comm, recvbuf, sendbuf, count * sizeof(T));
            }
        }
    }
    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void AllGather(T* buffer, size_t count, Communicator<GpushmemBackend>* comm) {
        AllGather<SCOPE>((buffer + comm->GlobalRank() * count), buffer, count, comm);
    }

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void AllGatherv(T* sendbuf, T* recvbuf, size_t* counts, size_t* displs,
                                             Communicator<GpushmemBackend>* comm) {
        comm->Barrier<SCOPE>();
        for (size_t i = 0; i < comm->GlobalSize(); i++) {
            if constexpr (SCOPE == ThreadGroup::BLOCK) {
                nvshmemx_putmem_nbi_block(recvbuf + displs[comm->GlobalRank()], sendbuf,
                                          counts[comm->GlobalRank()] * sizeof(T),
                                          nvshmem_team_translate_pe(comm->nvshmem_comm, i, NVSHMEM_TEAM_WORLD));
            } else if constexpr (SCOPE == ThreadGroup::WARP) {
                nvshmemx_putmem_nbi_warp(recvbuf + displs[comm->GlobalRank()], sendbuf,
                                         counts[comm->GlobalRank()] * sizeof(T),
                                         nvshmem_team_translate_pe(comm->nvshmem_comm, i, NVSHMEM_TEAM_WORLD));
            } else if constexpr (SCOPE == ThreadGroup::THREAD) {
                nvshmem_putmem_nbi(recvbuf + displs[comm->GlobalRank()], sendbuf,
                                   counts[comm->GlobalRank()] * sizeof(T),
                                   nvshmem_team_translate_pe(comm->nvshmem_comm, i, NVSHMEM_TEAM_WORLD));
            }
        }
        comm->Barrier<SCOPE>();
    }
    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void AllGatherv(T* buffer, size_t* counts, size_t* displs,
                                             Communicator<GpushmemBackend>* comm) {
        AllGatherv<SCOPE>((buffer + displs[comm->GlobalRank()]), buffer, counts, displs, comm);
    }

    template <ThreadGroup SCOPE, ReductionOperator OP, typename T>
    GPU_DEVICE static void AllReduce(const T* sendbuf, T* recvbuf, size_t count, Communicator<GpushmemBackend>* comm);
    template <ThreadGroup SCOPE, ReductionOperator OP, typename T>
    GPU_DEVICE static void AllReduce(T* buffer, size_t count, Communicator<GpushmemBackend>* comm);

    template <ThreadGroup SCOPE, ReductionOperator OP, typename T>
    GPU_DEVICE static void Reduce(const T* sendbuf, T* recvbuf, size_t count, int root,
                                  Communicator<GpushmemBackend>* comm);
    template <ThreadGroup SCOPE, ReductionOperator OP, typename T>
    GPU_DEVICE static void Reduce(T* buffer, size_t count, int root, Communicator<GpushmemBackend>* comm);

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void AlltoAll(const T* sendbuf, T* recvbuf, size_t count,
                                           Communicator<GpushmemBackend>* comm) {
        if constexpr (M == LaunchMode::FullDevice) {
            if constexpr (SCOPE == ThreadGroup::BLOCK) {
                nvshmemx_alltoallmem_block(comm->nvshmem_comm, recvbuf, sendbuf, count * sizeof(T));
            } else if constexpr (SCOPE == ThreadGroup::WARP) {
                nvshmemx_alltoallmem_warp(comm->nvshmem_comm, recvbuf, sendbuf, count * sizeof(T));
            } else if constexpr (SCOPE == ThreadGroup::THREAD) {
                nvshmem_alltoallmem(comm->nvshmem_comm, recvbuf, sendbuf, count * sizeof(T));
            }
        }
    }
    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void AlltoAllv(const T* sendbuf, size_t* send_counts, size_t* send_displs, T* recvbuf,
                                            size_t* recv_counts, size_t* recv_displs,
                                            Communicator<GpushmemBackend>* comm) {
        if constexpr (M == LaunchMode::FullDevice) {
            comm->Barrier<SCOPE>();
            for (size_t i = 0; i < comm->GlobalSize(); i++) {
                if constexpr (SCOPE == ThreadGroup::BLOCK) {
                    nvshmemx_putmem_nbi_block(recvbuf + recv_displs[comm->GlobalRank()],
                                              sendbuf + send_displs[comm->GlobalRank()],
                                              send_counts[comm->GlobalRank()] * sizeof(T),
                                              nvshmem_team_translate_pe(comm->nvshmem_comm, i, NVSHMEM_TEAM_WORLD));
                } else if constexpr (SCOPE == ThreadGroup::WARP) {
                    nvshmemx_putmem_nbi_warp(recvbuf + recv_displs[comm->GlobalRank()],
                                             sendbuf + send_displs[comm->GlobalRank()],
                                             send_counts[comm->GlobalRank()] * sizeof(T),
                                             nvshmem_team_translate_pe(comm->nvshmem_comm, i, NVSHMEM_TEAM_WORLD));
                } else if constexpr (SCOPE == ThreadGroup::THREAD) {
                    nvshmem_putmem_nbi(recvbuf + recv_displs[comm->GlobalRank()],
                                       sendbuf + send_displs[comm->GlobalRank()],
                                       send_counts[comm->GlobalRank()] * sizeof(T),
                                       nvshmem_team_translate_pe(comm->nvshmem_comm, i, NVSHMEM_TEAM_WORLD));
                }
            }
            comm->Barrier<SCOPE>();
        }
    }

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Broadcast(T* buffer, size_t count, int root, Communicator<GpushmemBackend>* comm) {
        if constexpr (M == LaunchMode::FullDevice) {
            if constexpr (SCOPE == ThreadGroup::BLOCK) {
                nvshmemx_broadcastmem_block(comm->nvshmem_comm, buffer, buffer, count,
                                            nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD));
            } else if constexpr (SCOPE == ThreadGroup::WARP) {
                nvshmemx_broadcastmem_warp(comm->nvshmem_comm, buffer, buffer, count,
                                           nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD));
            } else if constexpr (SCOPE == ThreadGroup::THREAD) {
                nvshmem_broadcastmem(comm->nvshmem_comm, buffer, buffer, count,
                                     nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD));
            }
        }
    }

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Gather(const T* sendbuf, T* recvbuf, size_t count, int root,
                                         Communicator<GpushmemBackend>* comm) {
        if constexpr (M == LaunchMode::FullDevice) {
            comm->Barrier<SCOPE>();
            if constexpr (SCOPE == ThreadGroup::BLOCK) {
                nvshmemx_putmem_nbi_block(recvbuf + comm->GlobalRank() * count, sendbuf, count * sizeof(T),
                                          nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD));
            } else if constexpr (SCOPE == ThreadGroup::WARP) {
                nvshmemx_putmem_nbi_warp(recvbuf + comm->GlobalRank() * count, sendbuf, count * sizeof(T),
                                         nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD));
            } else if constexpr (SCOPE == ThreadGroup::THREAD) {
                nvshmem_putmem_nbi(recvbuf + comm->GlobalRank() * count, sendbuf, count * sizeof(T),
                                   nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD));
            }
            comm->Barrier<SCOPE>();
        }
    }
    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Gather(T* buffer, size_t count, int root, Communicator<GpushmemBackend>* comm) {
        Gather<SCOPE>((buffer + comm->GlobalRank() * count), buffer, count, root, comm);
    }

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Gatherv(const T* sendbuf, T* recvbuf, size_t* counts, size_t* displs, int root,
                                          Communicator<GpushmemBackend>* comm) {
        if constexpr (M == LaunchMode::FullDevice) {
            comm->Barrier<SCOPE>();
            if constexpr (SCOPE == ThreadGroup::BLOCK) {
                nvshmemx_putmem_nbi_block(recvbuf + displs[comm->GlobalRank()], sendbuf,
                                          counts[comm->GlobalRank()] * sizeof(T),
                                          nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD));
            } else if constexpr (SCOPE == ThreadGroup::WARP) {
                nvshmemx_putmem_nbi_warp(recvbuf + displs[comm->GlobalRank()], sendbuf,
                                         counts[comm->GlobalRank()] * sizeof(T),
                                         nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD));
            } else if constexpr (SCOPE == ThreadGroup::THREAD) {
                nvshmem_putmem_nbi(recvbuf + displs[comm->GlobalRank()], sendbuf,
                                   counts[comm->GlobalRank()] * sizeof(T),
                                   nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD));
            }
            comm->Barrier<SCOPE>();
        }
    }
    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Gatherv(T* buffer, size_t* counts, size_t* displs, int root,
                                          Communicator<GpushmemBackend>* comm) {
        Gatherv<SCOPE>((buffer + displs[comm->GlobalRank()]), buffer, counts, displs, root, comm);
    }

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Scatter(const T* sendbuf, T* recvbuf, size_t count, int root,
                                          Communicator<GpushmemBackend>* comm) {
        if constexpr (M == LaunchMode::FullDevice) {
            comm->Barrier<SCOPE>();
            if constexpr (SCOPE == ThreadGroup::BLOCK) {
                nvshmemx_getmem_nbi_block(recvbuf, sendbuf + comm->GlobalRank() * count, count * sizeof(T),
                                          nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD));
            } else if constexpr (SCOPE == ThreadGroup::WARP) {
                nvshmemx_getmem_nbi_warp(recvbuf, sendbuf + comm->GlobalRank() * count, count * sizeof(T),
                                         nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD));
            } else if constexpr (SCOPE == ThreadGroup::THREAD) {
                nvshmem_getmem_nbi(recvbuf, sendbuf + comm->GlobalRank() * count, count * sizeof(T),
                                   nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD));
            }
            comm->Barrier<SCOPE>();
        }
    }
    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Scatter(T* buffer, size_t count, int root, Communicator<GpushmemBackend>* comm) {
        Scatter<SCOPE>((buffer + comm->GlobalRank() * count), buffer, count, root, comm);
    }

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Scatterv(const T* sendbuf, T* recvbuf, size_t* counts, size_t* displs, int root,
                                           Communicator<GpushmemBackend>* comm) {
        if constexpr (M == LaunchMode::FullDevice) {
            comm->Barrier<SCOPE>();
            if constexpr (SCOPE == ThreadGroup::BLOCK) {
                nvshmemx_getmem_nbi_block(recvbuf, sendbuf + displs[comm->GlobalRank()],
                                          counts[comm->GlobalRank()] * sizeof(T),
                                          nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD));
            } else if constexpr (SCOPE == ThreadGroup::WARP) {
                nvshmemx_getmem_nbi_warp(recvbuf, sendbuf + displs[comm->GlobalRank()],
                                         counts[comm->GlobalRank()] * sizeof(T),
                                         nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD));
            } else if constexpr (SCOPE == ThreadGroup::THREAD) {
                nvshmem_getmem_nbi(recvbuf, sendbuf + displs[comm->GlobalRank()],
                                   counts[comm->GlobalRank()] * sizeof(T),
                                   nvshmem_team_translate_pe(comm->nvshmem_comm, root, NVSHMEM_TEAM_WORLD));
            }
            comm->Barrier<SCOPE>();
        }
    }

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Scatterv(T* buffer, size_t* counts, size_t* displs, int root,
                                           Communicator<GpushmemBackend>* comm) {
        Scatterv<SCOPE>((buffer + displs[comm->GlobalRank()]), buffer, counts, displs, root, comm);
    }

    template <ThreadGroup SCOPE>
    GPU_DEVICE static inline void Wait();
    GPU_HOST ~Coordinator();
};

}  // namespace uniconn

// #include "operations.hpp"
#endif  // __UNICONN_INCLUDE_UNICONN_GPUSHMEM_COORDINATOR_HPP_
