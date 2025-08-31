#ifndef __UNICONN_INCLUDE_UNICONN_GPUCCL_COORDINATOR_HPP_
#define __UNICONN_INCLUDE_UNICONN_GPUCCL_COORDINATOR_HPP_
#include "common.hpp"
#include "communicator.hpp"
#include "uniconn/interfaces/coordinator.hpp"
namespace uniconn {

template <>
class Coordinator<GpucclBackend, LaunchMode::HostDriven> {
   private:
    UncGpuStream_t stream;
    KernelInfo kernel_info;

   public:
    GPU_HOST Coordinator(UncGpuStream_t stream);

    template <LaunchMode MODE>
    GPU_HOST void bindKernel(void* kernel_func, dim3 grid_dims, dim3 block_dims, size_t shared_mem, void** kernel_args);

    GPU_HOST inline void LaunchKernel() { kernel_info.launch(stream); }

    GPU_HOST inline void CommStart() { NCCL_CALL(ncclGroupStart()); }

    GPU_HOST inline void CommEnd() { NCCL_CALL(ncclGroupEnd()); }

    template <typename T>
    GPU_HOST inline void Post(T* src_buffer, T* dest_buffer, size_t buffer_size, uint64_t* signal_location,
                              uint64_t signal_val, int dest_process_id, Communicator<GpucclBackend>* comm) {
        NCCL_CALL(ncclSend((const void*)src_buffer, buffer_size, internal::gpuccl::TypeMap<T>(), dest_process_id,
                           comm->nccl_comm, stream));
    }

    template <typename T>
    GPU_HOST inline void Acknowledge(T* dest_buffer, size_t buffer_size, uint64_t* signal_location, uint64_t signal_val,
                                     int src_process_id, Communicator<GpucclBackend>* comm) {
        NCCL_CALL(ncclRecv((void*)dest_buffer, buffer_size, internal::gpuccl::TypeMap<T>(), src_process_id,
                           comm->nccl_comm, stream));
    }

    template <typename T>
    GPU_HOST inline void AllGather(T* sendbuf, T* recvbuf, size_t count, Communicator<GpucclBackend>* comm) {
        NCCL_CALL(ncclAllGather((const void*)sendbuf, (void*)recvbuf, count, internal::gpuccl::TypeMap<T>(),
                                comm->nccl_comm, stream));
    }
    template <typename T>
    GPU_HOST inline void AllGather(T* buffer, size_t count, Communicator<GpucclBackend>* comm) {
        AllGather((buffer + comm->GlobalRank() * count), buffer, count, comm);
    }

    template <typename T>
    GPU_HOST inline void AllGatherv(T* sendbuf, T* recvbuf, size_t* counts, size_t* displs,
                                    Communicator<GpucclBackend>* comm) {
        NCCL_CALL(ncclGroupStart());
        for (size_t i = 0; i < comm->GlobalSize(); ++i) {
            if (counts[i] > 0) {
                NCCL_CALL(ncclSend((const void*)sendbuf, counts[comm->GlobalRank()], internal::gpuccl::TypeMap<T>(), i,
                                   comm->nccl_comm, stream));
                NCCL_CALL(ncclRecv((void*)(recvbuf + displs[i]), counts[i], internal::gpuccl::TypeMap<T>(), i,
                                   comm->nccl_comm, stream));
            }
        }
        NCCL_CALL(ncclGroupEnd());
    }
    template <typename T>
    GPU_HOST inline void AllGatherv(T* buffer, size_t* counts, size_t* displs, Communicator<GpucclBackend>* comm) {
        AllGatherv((buffer + displs[comm->GlobalRank()]), buffer, counts, displs, comm);
    }

    template <ReductionOperator OP, typename T>
    GPU_HOST inline void AllReduce(const T* sendbuf, T* recvbuf, size_t count, Communicator<GpucclBackend>* comm) {
        NCCL_CALL(ncclAllReduce((const void*)sendbuf, (void*)recvbuf, count, internal::gpuccl::TypeMap<T>(),
                                internal::gpuccl::ReductOp2ncclRedOp<OP>(), comm->nccl_comm, stream));
    }
    template <ReductionOperator OP, typename T>
    GPU_HOST inline void AllReduce(T* buffer, size_t count, Communicator<GpucclBackend>* comm) {
        AllReduce<OP>(buffer, buffer, count, comm);
    }

    template <ReductionOperator OP, typename T>
    GPU_HOST inline void Reduce(const T* sendbuf, T* recvbuf, size_t count, int root,
                                Communicator<GpucclBackend>* comm) {
        NCCL_CALL(ncclReduce((const void*)sendbuf, (void*)recvbuf, count, internal::gpuccl::TypeMap<T>(),
                             internal::gpuccl::ReductOp2ncclRedOp<OP>(), root, comm->nccl_comm, stream));
    }
    template <ReductionOperator OP, typename T>
    GPU_HOST inline void Reduce(T* buffer, size_t count, int root, Communicator<GpucclBackend>* comm) {
        Reduce<OP>(buffer, buffer, count, root, comm);
    }

    template <typename T>
    GPU_HOST inline void AlltoAll(const T* sendbuf, T* recvbuf, size_t count, Communicator<GpucclBackend>* comm) {
        NCCL_CALL(ncclGroupStart());
        for (size_t i = 0; i < comm->GlobalSize(); ++i) {
            NCCL_CALL(ncclSend((const void*)&sendbuf[i * count], count, internal::gpuccl::TypeMap<T>(), i,
                               comm->nccl_comm, stream));
            NCCL_CALL(ncclRecv((void*)&recvbuf[i * count], count, internal::gpuccl::TypeMap<T>(), i, comm->nccl_comm,
                               stream));
        }
        NCCL_CALL(ncclGroupEnd());
    }
    template <typename T>
    GPU_HOST inline void AlltoAllv(const T* sendbuf, size_t* send_counts, size_t* send_displs, T* recvbuf,
                                   size_t* recv_counts, size_t* recv_displs, Communicator<GpucclBackend>* comm) {
        NCCL_CALL(ncclGroupStart());
        for (size_t i = 0; i < comm->GlobalSize(); ++i) {
            if (recv_counts[i] > 0) {
                NCCL_CALL(ncclRecv((void*)(recvbuf + recv_displs[i]), recv_counts[i], internal::gpuccl::TypeMap<T>(), i,
                                   comm->nccl_comm, stream));
            }

            if (send_counts[i] > 0) {
                NCCL_CALL(ncclSend((const void*)(sendbuf + send_displs[i]), send_counts[i],
                                   internal::gpuccl::TypeMap<T>(), i, comm->nccl_comm, stream));
            }
        }
        NCCL_CALL(ncclGroupEnd());
    }

    template <typename T>
    GPU_HOST inline void Broadcast(T* buffer, size_t count, int root, Communicator<GpucclBackend>* comm) {
        NCCL_CALL(ncclBroadcast((const void*)buffer, (void*)buffer, count, internal::gpuccl::TypeMap<T>(), root,
                                comm->nccl_comm, stream));
    }

    template <typename T>
    GPU_HOST inline void Gather(const T* sendbuf, T* recvbuf, size_t count, int root,
                                Communicator<GpucclBackend>* comm) {
        NCCL_CALL(ncclGroupStart());
        NCCL_CALL(ncclSend((const void*)sendbuf, count, internal::gpuccl::TypeMap<T>(), root, comm->nccl_comm, stream));
        if (comm->GlobalRank() == root) {
            for (size_t i = 0; i < comm->GlobalSize(); ++i) {
                NCCL_CALL(ncclRecv((void*)&recvbuf[i * count], count, internal::gpuccl::TypeMap<T>(), i,
                                   comm->nccl_comm, stream));
            }
        }
        NCCL_CALL(ncclGroupEnd());
    }
    template <typename T>
    GPU_HOST inline void Gather(T* buffer, size_t count, int root, Communicator<GpucclBackend>* comm) {
        Gather((buffer + comm->GlobalRank() * count), buffer, count, root, comm);
    }

    template <typename T>
    GPU_HOST inline void Gatherv(const T* sendbuf, T* recvbuf, size_t* counts, size_t* displs, int root,
                                 Communicator<GpucclBackend>* comm) {
        NCCL_CALL(ncclGroupStart());
        if (counts[comm->GlobalRank()] != 0) {
            NCCL_CALL(ncclSend((const void*)sendbuf, counts[comm->GlobalRank()], internal::gpuccl::TypeMap<T>(), root,
                               comm->nccl_comm, stream));
        }

        if (comm->GlobalRank() == root) {
            for (size_t i = 0; i < comm->GlobalSize(); ++i) {
                if (counts[i] != 0) {
                    NCCL_CALL(ncclRecv((void*)(recvbuf + displs[i]), counts[i], internal::gpuccl::TypeMap<T>(), i,
                                       comm->nccl_comm, stream));
                }
            }
        }
        NCCL_CALL(ncclGroupEnd());
    }
    template <typename T>
    GPU_HOST inline void Gatherv(T* buffer, size_t* counts, size_t* displs, int root,
                                 Communicator<GpucclBackend>* comm) {
        Gatherv((buffer + displs[comm->GlobalRank()]), buffer, counts, displs, root, comm);
    }

    template <typename T>
    GPU_HOST inline void Scatter(const T* sendbuf, T* recvbuf, size_t count, int root,
                                 Communicator<GpucclBackend>* comm) {
        NCCL_CALL(ncclGroupStart());

        NCCL_CALL(ncclRecv((void*)recvbuf, count, internal::gpuccl::TypeMap<T>(), root, comm->nccl_comm, stream));
        if (comm->GlobalRank() == root) {
            for (size_t i = 0; i < comm->GlobalSize(); ++i) {
                NCCL_CALL(ncclSend((const void*)&sendbuf[i * count], count, internal::gpuccl::TypeMap<T>(), i,
                                   comm->nccl_comm, stream));
            }
        }
        NCCL_CALL(ncclGroupEnd());
    }
    template <typename T>
    GPU_HOST inline void Scatter(T* buffer, size_t count, int root, Communicator<GpucclBackend>* comm) {
        Scatter((buffer + comm->GlobalRank() * count), buffer, count, root, comm);
    }

    template <typename T>
    GPU_HOST inline void Scatterv(const T* sendbuf, T* recvbuf, size_t* counts, size_t* displs, int root,
                                  Communicator<GpucclBackend>* comm) {
        NCCL_CALL(ncclGroupStart());
        if (counts[comm->GlobalRank()] != 0) {
            NCCL_CALL(ncclRecv((void*)recvbuf, counts[comm->GlobalRank()], internal::gpuccl::TypeMap<T>(), root,
                               comm->nccl_comm, stream));
        }

        if (comm->GlobalRank() == root) {
            for (size_t i = 0; i < comm->GlobalSize(); ++i) {
                if (counts[i] != 0) {
                    NCCL_CALL(ncclSend((const void*)(sendbuf + displs[i]), counts[i], internal::gpuccl::TypeMap<T>(), i,
                                       comm->nccl_comm, stream));
                }
            }
        }
        NCCL_CALL(ncclGroupEnd());
    }

    template <typename T>
    GPU_HOST inline void Scatterv(T* buffer, size_t* counts, size_t* displs, int root,
                                  Communicator<GpucclBackend>* comm) {
        Scatterv((buffer + displs[comm->GlobalRank()]), buffer, counts, displs, root, comm);
    }

    GPU_HOST inline void WaitComm() {}

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Post(T* src_buffer, T* dest_buffer, size_t buffer_size, uint64_t* signal_location,
                                       uint64_t signal_val, int dest_process_id, Communicator<GpucclBackend>* comm) {}

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Acknowledge(T* dest_buffer, size_t buffer_size, uint64_t* signal_location,
                                              uint64_t signal_val, int src_process_id,
                                              Communicator<GpucclBackend>* comm) {}

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void AllGather(T* sendbuf, T* recvbuf, size_t count, Communicator<GpucclBackend>* comm) {}
    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void AllGather(T* buffer, size_t count, Communicator<GpucclBackend>* comm) {}

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void AllGatherv(T* sendbuf, T* recvbuf, size_t* counts, size_t* displs,
                                             Communicator<GpucclBackend>* comm) {}
    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void AllGatherv(T* buffer, size_t* counts, size_t* displs,
                                             Communicator<GpucclBackend>* comm) {}

    template <ThreadGroup SCOPE, ReductionOperator OP, typename T>
    GPU_DEVICE static inline void AllReduce(const T* sendbuf, T* recvbuf, size_t count,
                                            Communicator<GpucclBackend>* comm) {}
    template <ThreadGroup SCOPE, ReductionOperator OP, typename T>
    GPU_DEVICE static inline void AllReduce(T* buffer, size_t count, Communicator<GpucclBackend>* comm) {}

    template <ThreadGroup SCOPE, ReductionOperator OP, typename T>
    GPU_DEVICE static inline void Reduce(const T* sendbuf, T* recvbuf, size_t count, int root,
                                         Communicator<GpucclBackend>* comm) {}
    template <ThreadGroup SCOPE, ReductionOperator OP, typename T>
    GPU_DEVICE static inline void Reduce(T* buffer, size_t count, int root, Communicator<GpucclBackend>* comm) {}

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void AlltoAll(const T* sendbuf, T* recvbuf, size_t count,
                                           Communicator<GpucclBackend>* comm) {}
    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void AlltoAllv(const T* sendbuf, size_t* send_counts, size_t* send_displs, T* recvbuf,
                                            size_t* recv_counts, size_t* recv_displs,
                                            Communicator<GpucclBackend>* comm) {}

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Broadcast(T* buffer, size_t count, int root, Communicator<GpucclBackend>* comm) {}

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Gather(const T* sendbuf, T* recvbuf, size_t count, int root,
                                         Communicator<GpucclBackend>* comm) {}
    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Gather(T* buffer, size_t count, int root, Communicator<GpucclBackend>* comm) {}

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Gatherv(const T* sendbuf, T* recvbuf, size_t* counts, size_t* displs, int root,
                                          Communicator<GpucclBackend>* comm) {}
    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Gatherv(T* buffer, size_t* counts, size_t* displs, int root,
                                          Communicator<GpucclBackend>* comm) {}

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Scatter(const T* sendbuf, T* recvbuf, size_t count, int root,
                                          Communicator<GpucclBackend>* comm) {}
    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Scatter(T* buffer, size_t count, int root, Communicator<GpucclBackend>* comm) {}

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Scatterv(const T* sendbuf, T* recvbuf, size_t* counts, size_t* displs, int root,
                                           Communicator<GpucclBackend>* comm) {}

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Scatterv(T* buffer, size_t* counts, size_t* displs, int root,
                                           Communicator<GpucclBackend>* comm) {}
    template <ThreadGroup SCOPE>
    GPU_DEVICE static inline void Wait() {}
    GPU_HOST ~Coordinator();
    // GPU_HOST Coordinator(UncGpuStream_t stream) : stream(stream), comm_count(0), kernel_info() {}

    // template <LaunchMode MODE>
    // GPU_HOST void bindKernel(void* kernel_func, dim3 grid_dims, dim3 block_dims, size_t shared_mem,
    //                          void** kernel_args) {
    //     if constexpr (MODE == LaunchMode::HostDriven) {
    //         kernel_info = KernelInfo(kernel_func, kernel_args, grid_dims, block_dims, shared_mem);
    //     }
    // }

    // GPU_HOST void LaunchKernel() { kernel_info.launch(stream); }

    // GPU_HOST void CommStart() { NCCL_CALL(ncclGroupStart()); }

    // GPU_HOST void CommEnd() {
    //     NCCL_CALL(ncclGroupEnd());
    //     comm_count = 0;
    // }

    // template <typename T>
    // GPU_HOST void Post(T* src_buffer, T* dest_buffer, size_t buffer_size, uint64_t* signal_location,
    //                    uint64_t signal_val, int dest_process_id, Communicator<GpucclBackend>* comm) {
    //     comm_count++;
    //     if ((comm_count & 1023) == 0) {
    //         NCCL_CALL(ncclGroupEnd());
    //         NCCL_CALL(ncclGroupStart());
    //     }
    //     NCCL_CALL(ncclSend((const void*)src_buffer, buffer_size, internal::gpuccl::TypeMap<T>(), dest_process_id,
    //                        comm->nccl_comm, stream));
    // }

    // template <typename T>
    // GPU_HOST void Acknowledge(T* dest_buffer, size_t buffer_size, uint64_t* signal_location, uint64_t signal_val,
    //                           int src_process_id, Communicator<GpucclBackend>* comm) {
    //     comm_count++;
    //     if ((comm_count & 1023) == 0) {
    //         NCCL_CALL(ncclGroupEnd());
    //         NCCL_CALL(ncclGroupStart());
    //     }
    //     NCCL_CALL(ncclRecv((void*)dest_buffer, buffer_size, internal::gpuccl::TypeMap<T>(), src_process_id,
    //                        comm->nccl_comm, stream));
    // }

    // template <typename T>
    // GPU_HOST void AllGather(T* sendbuf, T* recvbuf, size_t count, Communicator<GpucclBackend>* comm) {
    //     comm_count++;
    //     if ((comm_count & UNC_MAX_GPUCCL_CALL_PER_GROUP) == 0) {
    //         NCCL_CALL(ncclGroupEnd());
    //         NCCL_CALL(ncclGroupStart());
    //     }
    //     NCCL_CALL(ncclAllGather((const void*)sendbuf, (void*)recvbuf, count, internal::gpuccl::TypeMap<T>(),
    //                             comm->nccl_comm, stream));
    // }
    // template <typename T>
    // GPU_HOST void AllGather(T* buffer, size_t count, Communicator<GpucclBackend>* comm) {
    //     AllGather((buffer + comm->GlobalRank() * count), buffer, count, comm);
    // }

    // template <typename T>
    // GPU_HOST void AllGatherv(T* sendbuf, T* recvbuf, size_t* counts, size_t* displs,
    //                          Communicator<GpucclBackend>* comm) {
    //     for (size_t i = 0; i < comm->GlobalSize(); ++i) {
    //         comm_count += (counts[i] > 0) ? 2 : 0;
    //         if ((comm_count & 1023) == 0) {
    //             NCCL_CALL(ncclGroupEnd());
    //             NCCL_CALL(ncclGroupStart());
    //         }

    //         if (counts[i] > 0) {
    //             NCCL_CALL(ncclRecv((void*)(recvbuf + displs[i]), counts[i], internal::gpuccl::TypeMap<T>(), i,
    //                                comm->nccl_comm, stream));
    //             NCCL_CALL(ncclSend((const void*)sendbuf, counts[comm->GlobalRank()], internal::gpuccl::TypeMap<T>(),
    //             i,
    //                                comm->nccl_comm, stream));
    //         }
    //     }
    // }
    // template <typename T>
    // GPU_HOST void AllGatherv(T* buffer, size_t* counts, size_t* displs, Communicator<GpucclBackend>* comm) {
    //     AllGatherv((buffer + displs[comm->GlobalRank()]), buffer, counts, displs, comm);
    // }

    // template <ReductionOperator OP, typename T>
    // GPU_HOST void AllReduce(const T* sendbuf, T* recvbuf, size_t count, Communicator<GpucclBackend>* comm) {
    //     comm_count++;
    //     if ((comm_count & 1023) == 0) {
    //         NCCL_CALL(ncclGroupEnd());
    //         NCCL_CALL(ncclGroupStart());
    //     }
    //     NCCL_CALL(ncclAllReduce((const void*)sendbuf, (void*)recvbuf, count, internal::gpuccl::TypeMap<T>(),
    //                             internal::gpuccl::ReductOp2ncclRedOp<OP>(), comm->nccl_comm, stream));
    // }
    // template <ReductionOperator OP, typename T>
    // GPU_HOST void AllReduce(T* buffer, size_t count, Communicator<GpucclBackend>* comm) {
    //     AllReduce<OP>(buffer, buffer, count, comm);
    // }

    // template <ReductionOperator OP, typename T>
    // GPU_HOST void Reduce(const T* sendbuf, T* recvbuf, size_t count, int root, Communicator<GpucclBackend>* comm) {
    //     comm_count++;
    //     if ((comm_count & 1023) == 0) {
    //         NCCL_CALL(ncclGroupEnd());
    //         NCCL_CALL(ncclGroupStart());
    //     }
    //     NCCL_CALL(ncclReduce((const void*)sendbuf, (void*)recvbuf, count, internal::gpuccl::TypeMap<T>(),
    //                          internal::gpuccl::ReductOp2ncclRedOp<OP>(), root, comm->nccl_comm, stream));
    // }
    // template <ReductionOperator OP, typename T>
    // GPU_HOST void Reduce(T* buffer, size_t count, int root, Communicator<GpucclBackend>* comm) {
    //     Reduce<OP>(buffer, buffer, count, root, comm);
    // }

    // template <typename T>
    // GPU_HOST void AlltoAll(const T* sendbuf, T* recvbuf, size_t count, Communicator<GpucclBackend>* comm) {
    //     for (size_t i = 0; i < comm->GlobalSize(); ++i) {
    //         comm_count += 2;
    //         if ((comm_count & 1023) == 0) {
    //             NCCL_CALL(ncclGroupEnd());
    //             NCCL_CALL(ncclGroupStart());
    //         }

    //         NCCL_CALL(ncclSend((const void*)&sendbuf[i * count], count, internal::gpuccl::TypeMap<T>(), i,
    //                            comm->nccl_comm, stream));
    //         NCCL_CALL(ncclRecv((void*)&recvbuf[i * count], count, internal::gpuccl::TypeMap<T>(), i, comm->nccl_comm,
    //                            stream));
    //     }
    // }
    // template <typename T>
    // GPU_HOST void AlltoAllv(const T* sendbuf, size_t* send_counts, size_t* send_displs, T* recvbuf, size_t*
    // recv_counts,
    //                         size_t* recv_displs, Communicator<GpucclBackend>* comm) {
    //     for (size_t i = 0; i < comm->GlobalSize(); ++i) {
    //         if (recv_counts[i] > 0) {
    //             comm_count++;
    //             if ((comm_count & 1023) == 0) {
    //                 NCCL_CALL(ncclGroupEnd());
    //                 NCCL_CALL(ncclGroupStart());
    //             }
    //             NCCL_CALL(ncclRecv((void*)(recvbuf + recv_displs[i]), recv_counts[i], internal::gpuccl::TypeMap<T>(),
    //             i,
    //                                comm->nccl_comm, stream));
    //         }

    //         if (send_counts[i] > 0) {
    //             comm_count++;
    //             if ((comm_count & 1023) == 0) {
    //                 NCCL_CALL(ncclGroupEnd());
    //                 NCCL_CALL(ncclGroupStart());
    //             }
    //             NCCL_CALL(ncclSend((const void*)(sendbuf + send_displs[i]), send_counts[i],
    //                                internal::gpuccl::TypeMap<T>(), i, comm->nccl_comm, stream));
    //         }
    //     }
    // }

    // template <typename T>
    // GPU_HOST void Broadcast(T* buffer, size_t count, int root, Communicator<GpucclBackend>* comm) {
    //     comm_count++;
    //     if ((comm_count & 1023) == 0) {
    //         NCCL_CALL(ncclGroupEnd());
    //         NCCL_CALL(ncclGroupStart());
    //     }
    //     NCCL_CALL(ncclBroadcast((const void*)buffer, (void*)buffer, count, internal::gpuccl::TypeMap<T>(), root,
    //                             comm->nccl_comm, stream));
    // }

    // template <typename T>
    // GPU_HOST void Gather(const T* sendbuf, T* recvbuf, size_t count, int root, Communicator<GpucclBackend>* comm) {
    //     comm_count++;
    //     if ((comm_count & 1023) == 0) {
    //         NCCL_CALL(ncclGroupEnd());
    //         NCCL_CALL(ncclGroupStart());
    //     }
    //     NCCL_CALL(ncclSend((const void*)sendbuf, count, internal::gpuccl::TypeMap<T>(), root, comm->nccl_comm,
    //     stream)); if (comm->GlobalRank() == root) {
    //         for (size_t i = 0; i < comm->GlobalSize(); ++i) {
    //             comm_count++;
    //             if ((comm_count & 1023) == 0) {
    //                 NCCL_CALL(ncclGroupEnd());
    //                 NCCL_CALL(ncclGroupStart());
    //             }

    //             NCCL_CALL(ncclRecv((void*)&recvbuf[i * count], count, internal::gpuccl::TypeMap<T>(), i,
    //                                comm->nccl_comm, stream));
    //         }
    //     }
    // }
    // template <typename T>
    // GPU_HOST void Gather(T* buffer, size_t count, int root, Communicator<GpucclBackend>* comm) {
    //     Gather((buffer + comm->GlobalRank() * count), buffer, count, root, comm);
    // }

    // template <typename T>
    // GPU_HOST void Gatherv(const T* sendbuf, T* recvbuf, size_t* counts, size_t* displs, int root,
    //                       Communicator<GpucclBackend>* comm) {
    //     if (counts[comm->GlobalRank()] != 0) {
    //         comm_count++;
    //         if ((comm_count & 1023) == 0) {
    //             NCCL_CALL(ncclGroupEnd());
    //             NCCL_CALL(ncclGroupStart());
    //         }
    //         NCCL_CALL(ncclSend((const void*)sendbuf, counts[comm->GlobalRank()], internal::gpuccl::TypeMap<T>(),
    //         root,
    //                            comm->nccl_comm, stream));
    //     }

    //     if (comm->GlobalRank() == root) {
    //         for (size_t i = 0; i < comm->GlobalSize(); ++i) {
    //             if (counts[i] != 0) {
    //                 comm_count++;
    //                 if ((comm_count & 1023) == 0) {
    //                     NCCL_CALL(ncclGroupEnd());
    //                     NCCL_CALL(ncclGroupStart());
    //                 }
    //                 NCCL_CALL(ncclRecv((void*)(recvbuf + displs[i]), counts[i], internal::gpuccl::TypeMap<T>(), i,
    //                                    comm->nccl_comm, stream));
    //             }
    //         }
    //     }
    // }
    // template <typename T>
    // GPU_HOST void Gatherv(T* buffer, size_t* counts, size_t* displs, int root, Communicator<GpucclBackend>* comm) {
    //     Gatherv((buffer + displs[comm->GlobalRank()]), buffer, counts, displs, root, comm);
    // }

    // template <typename T>
    // GPU_HOST void Scatter(const T* sendbuf, T* recvbuf, size_t count, int root, Communicator<GpucclBackend>* comm) {
    //     comm_count++;
    //     if ((comm_count & 1023) == 0) {
    //         NCCL_CALL(ncclGroupEnd());
    //         NCCL_CALL(ncclGroupStart());
    //     }
    //     NCCL_CALL(ncclRecv((void*)recvbuf, count, internal::gpuccl::TypeMap<T>(), root, comm->nccl_comm, stream));

    //     if (comm->GlobalRank() == root) {
    //         for (size_t i = 0; i < comm->GlobalSize(); ++i) {
    //             comm_count++;
    //             if ((comm_count & 1023) == 0) {
    //                 NCCL_CALL(ncclGroupEnd());
    //                 NCCL_CALL(ncclGroupStart());
    //             }
    //             NCCL_CALL(ncclSend((const void*)&sendbuf[i * count], count, internal::gpuccl::TypeMap<T>(), i,
    //                                comm->nccl_comm, stream));
    //         }
    //     }
    // }
    // template <typename T>
    // GPU_HOST void Scatter(T* buffer, size_t count, int root, Communicator<GpucclBackend>* comm) {
    //     Scatter((buffer + comm->GlobalRank() * count), buffer, count, root, comm);
    // }

    // template <typename T>
    // GPU_HOST void Scatterv(const T* sendbuf, T* recvbuf, size_t* counts, size_t* displs, int root,
    //                        Communicator<GpucclBackend>* comm) {
    //     if (counts[comm->GlobalRank()] != 0) {
    //         comm_count++;
    //         if ((comm_count & 1023) == 0) {
    //             NCCL_CALL(ncclGroupEnd());
    //             NCCL_CALL(ncclGroupStart());
    //         }
    //         NCCL_CALL(ncclRecv((void*)recvbuf, counts[comm->GlobalRank()], internal::gpuccl::TypeMap<T>(), root,
    //                            comm->nccl_comm, stream));
    //     }

    //     if (comm->GlobalRank() == root) {
    //         for (size_t i = 0; i < comm->GlobalSize(); ++i) {
    //             if (counts[i] != 0) {
    //                 comm_count++;
    //                 if ((comm_count & 1023) == 0) {
    //                     NCCL_CALL(ncclGroupEnd());
    //                     NCCL_CALL(ncclGroupStart());
    //                 }
    //                 NCCL_CALL(ncclSend((const void*)(sendbuf + displs[i]), counts[i], internal::gpuccl::TypeMap<T>(),
    //                 i,
    //                                    comm->nccl_comm, stream));
    //             }
    //         }
    //     }
    // }

    // template <typename T>
    // GPU_HOST void Scatterv(T* buffer, size_t* counts, size_t* displs, int root, Communicator<GpucclBackend>* comm) {
    //     Scatterv((buffer + displs[comm->GlobalRank()]), buffer, counts, displs, root, comm);
    // }

    // GPU_HOST void WaitComm() {}

    // template <ThreadGroup SCOPE, typename T>
    // GPU_DEVICE static void Post(T* src_buffer, T* dest_buffer, size_t buffer_size, uint64_t* signal_location,
    //                             uint64_t signal_val, int dest_process_id, Communicator<GpucclBackend>* comm) {}

    // template <ThreadGroup SCOPE, typename T>
    // GPU_DEVICE static void Acknowledge(T* dest_buffer, size_t buffer_size, uint64_t* signal_location,
    //                                    uint64_t signal_val, int src_process_id, Communicator<GpucclBackend>* comm) {}

    // template <ThreadGroup SCOPE, typename T>
    // GPU_DEVICE static void AllGather(T* sendbuf, T* recvbuf, size_t count, Communicator<GpucclBackend>* comm) {}
    // template <ThreadGroup SCOPE, typename T>
    // GPU_DEVICE static void AllGather(T* buffer, size_t count, Communicator<GpucclBackend>* comm) {}

    // template <ThreadGroup SCOPE, typename T>
    // GPU_DEVICE static void AllGatherv(T* sendbuf, T* recvbuf, size_t* counts, size_t* displs,
    //                                   Communicator<GpucclBackend>* comm) {}
    // template <ThreadGroup SCOPE, typename T>
    // GPU_DEVICE static void AllGatherv(T* buffer, size_t* counts, size_t* displs, Communicator<GpucclBackend>* comm)
    // {}

    // template <ThreadGroup SCOPE, ReductionOperator OP, typename T>
    // GPU_DEVICE static void AllReduce(const T* sendbuf, T* recvbuf, size_t count, Communicator<GpucclBackend>* comm)
    // {} template <ThreadGroup SCOPE, ReductionOperator OP, typename T> GPU_DEVICE static void AllReduce(T* buffer,
    // size_t count, Communicator<GpucclBackend>* comm) {}

    // template <ThreadGroup SCOPE, ReductionOperator OP, typename T>
    // GPU_DEVICE static void Reduce(const T* sendbuf, T* recvbuf, size_t count, int root,
    //                               Communicator<GpucclBackend>* comm) {}
    // template <ThreadGroup SCOPE, ReductionOperator OP, typename T>
    // GPU_DEVICE static void Reduce(T* buffer, size_t count, int root, Communicator<GpucclBackend>* comm) {}

    // template <ThreadGroup SCOPE, typename T>
    // GPU_DEVICE static void AlltoAll(const T* sendbuf, T* recvbuf, size_t count, Communicator<GpucclBackend>* comm) {}
    // template <ThreadGroup SCOPE, typename T>
    // GPU_DEVICE static void AlltoAllv(const T* sendbuf, size_t* send_counts, size_t* send_displs, T* recvbuf,
    //                                  size_t* recv_counts, size_t* recv_displs, Communicator<GpucclBackend>* comm) {}

    // template <ThreadGroup SCOPE, typename T>
    // GPU_DEVICE static void Broadcast(T* buffer, size_t count, int root, Communicator<GpucclBackend>* comm) {}

    // template <ThreadGroup SCOPE, typename T>
    // GPU_DEVICE static void Gather(const T* sendbuf, T* recvbuf, size_t count, int root,
    //                               Communicator<GpucclBackend>* comm) {}
    // template <ThreadGroup SCOPE, typename T>
    // GPU_DEVICE static void Gather(T* buffer, size_t count, int root, Communicator<GpucclBackend>* comm) {}

    // template <ThreadGroup SCOPE, typename T>
    // GPU_DEVICE static void Gatherv(const T* sendbuf, T* recvbuf, size_t* counts, size_t* displs, int root,
    //                                Communicator<GpucclBackend>* comm) {}
    // template <ThreadGroup SCOPE, typename T>
    // GPU_DEVICE static void Gatherv(T* buffer, size_t* counts, size_t* displs, int root,
    //                                Communicator<GpucclBackend>* comm) {}

    // template <ThreadGroup SCOPE, typename T>
    // GPU_DEVICE static void Scatter(const T* sendbuf, T* recvbuf, size_t count, int root,
    //                                Communicator<GpucclBackend>* comm) {}
    // template <ThreadGroup SCOPE, typename T>
    // GPU_DEVICE static void Scatter(T* buffer, size_t count, int root, Communicator<GpucclBackend>* comm) {}

    // template <ThreadGroup SCOPE, typename T>
    // GPU_DEVICE static void Scatterv(const T* sendbuf, T* recvbuf, size_t* counts, size_t* displs, int root,
    //                                 Communicator<GpucclBackend>* comm) {}

    // template <ThreadGroup SCOPE, typename T>
    // GPU_DEVICE static void Scatterv(T* buffer, size_t* counts, size_t* displs, int root,
    //                                 Communicator<GpucclBackend>* comm) {}
    // template <ThreadGroup SCOPE>
    // GPU_DEVICE static void Wait() {}
    // GPU_HOST ~Coordinator() {}
};

}  // namespace uniconn

#endif  // __UNICONN_INCLUDE_UNICONN_GPUCCL_COORDINATOR_HPP_
