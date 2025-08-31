#ifndef __UNICONN_INCLUDE_UNICONN_GPUMPI_COORDINATOR_HPP_
#define __UNICONN_INCLUDE_UNICONN_GPUMPI_COORDINATOR_HPP_
#include "common.hpp"
#include "communicator.hpp"
#include "uniconn/interfaces/coordinator.hpp"
namespace uniconn {
template <>
class Coordinator<MPIBackend, LaunchMode::HostDriven> {
   private:
    UncGpuStream_t stream;

    std::vector<MPI_Request> internal_reqs;
    std::vector<internal::mpi::Unc_mpi_count_t> temp_send_counts;
    std::vector<internal::mpi::Unc_mpi_count_t> temp_send_displs;
    std::vector<internal::mpi::Unc_mpi_count_t> temp_recv_counts;
    std::vector<internal::mpi::Unc_mpi_count_t> temp_recv_displs;
    KernelInfo kernel_info;
    uint64_t is_grouped = 0;

   public:
    GPU_HOST Coordinator(UncGpuStream_t stream);

    template <LaunchMode MODE>
    GPU_HOST void bindKernel(void* kernel_func, dim3 grid_dims, dim3 block_dims, size_t shared_mem, void** kernel_args);

    GPU_HOST inline void LaunchKernel() { kernel_info.launch(stream); }

    GPU_HOST inline void CommStart() {
        if (is_grouped == 0) {
            if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
                GPU_RT_CALL(UncGpuStreamSynchronize(stream));
            }
            is_grouped++;
        }
    }

    GPU_HOST inline void CommEnd() {
        if (is_grouped == 1) {
            is_grouped--;
            MPI_CALL(MPI_Waitall(internal_reqs.size(), internal_reqs.data(), MPI_STATUSES_IGNORE));
            internal_reqs.clear();
        }
    }

    template <typename T>
    GPU_HOST inline void Post(T* src_buffer, T* dest_buffer, size_t buffer_size, uint64_t* signal_location,
                              uint64_t signal_val, int dest_process_id, Communicator<MPIBackend>* comm) {
        if (is_grouped == 0) {
            if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
                GPU_RT_CALL(UncGpuStreamSynchronize(stream));
            }
            MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Send)(src_buffer, buffer_size, internal::mpi::TypeMap<T>(),
                                                        dest_process_id, 2, comm->mpi_comm));
        } else {
            MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Isend)(src_buffer, buffer_size, internal::mpi::TypeMap<T>(),
                                                         dest_process_id, 2, comm->mpi_comm,
                                                         &internal_reqs.emplace_back(MPI_REQUEST_NULL)));
        }
    }

    template <typename T>
    GPU_HOST inline void Acknowledge(T* dest_buffer, size_t buffer_size, uint64_t* signal_location, uint64_t signal_val,
                                     int src_process_id, Communicator<MPIBackend>* comm) {
        if (is_grouped == 0) {
            if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
                GPU_RT_CALL(UncGpuStreamSynchronize(stream));
            }
            MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Recv)(dest_buffer, buffer_size, internal::mpi::TypeMap<T>(),
                                                        src_process_id, 2, comm->mpi_comm, MPI_STATUS_IGNORE));
        } else {
            MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Irecv)(dest_buffer, buffer_size, internal::mpi::TypeMap<T>(),
                                                         src_process_id, 2, comm->mpi_comm,
                                                         &internal_reqs.emplace_back(MPI_REQUEST_NULL)));
        }
    }

    template <typename T>
    GPU_HOST inline void AllGather(T* sendbuf, T* recvbuf, size_t count, Communicator<MPIBackend>* comm) {
        if (is_grouped == 0) {
            if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
                GPU_RT_CALL(UncGpuStreamSynchronize(stream));
            }

            MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Allgather)(internal::mpi::buf_or_inplace(sendbuf), count,
                                                             internal::mpi::TypeMap<T>(), recvbuf, count,
                                                             internal::mpi::TypeMap<T>(), comm->mpi_comm));

        } else {
            MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Iallgather)(
                internal::mpi::buf_or_inplace(sendbuf), count, internal::mpi::TypeMap<T>(), recvbuf, count,
                internal::mpi::TypeMap<T>(), comm->mpi_comm, &internal_reqs.emplace_back(MPI_REQUEST_NULL)));
        }
    }
    template <typename T>
    GPU_HOST inline void AllGather(T* buffer, size_t count, Communicator<MPIBackend>* comm) {
        AllGather(internal::IN_PLACE<T>, buffer, count, comm);
    }

    template <typename T>
    GPU_HOST inline void AllGatherv(T* sendbuf, T* recvbuf, size_t* counts, size_t* displs,
                                    Communicator<MPIBackend>* comm) {
        {
            if (temp_recv_counts.empty() && temp_recv_displs.empty()) {
                for (size_t i = 0; i < comm->GlobalSize(); i++) {
                    temp_recv_counts.emplace_back(counts[i]);
                    temp_recv_displs.emplace_back(displs[i]);
                }
                if (is_grouped == 0) {
                    if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
                        GPU_RT_CALL(UncGpuStreamSynchronize(stream));
                    }
                    MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Allgatherv)(
                        internal::mpi::buf_or_inplace(sendbuf), counts[comm->GlobalRank()], internal::mpi::TypeMap<T>(),
                        recvbuf, temp_recv_counts.data(), temp_recv_displs.data(), internal::mpi::TypeMap<T>(),
                        comm->mpi_comm));

                    temp_recv_counts.clear();
                    temp_recv_displs.clear();
                } else {
                    MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Iallgatherv)(
                        internal::mpi::buf_or_inplace(sendbuf), counts[comm->GlobalRank()], internal::mpi::TypeMap<T>(),
                        recvbuf, temp_recv_counts.data(), temp_recv_displs.data(), internal::mpi::TypeMap<T>(),
                        comm->mpi_comm, &internal_reqs.emplace_back(MPI_REQUEST_NULL)));
                }
            }
        }
    }
    template <typename T>
    GPU_HOST inline void AllGatherv(T* buffer, size_t* counts, size_t* displs, Communicator<MPIBackend>* comm) {
        AllGatherv(internal::IN_PLACE<T>, buffer, counts, displs, comm);
    }

    template <ReductionOperator OP, typename T>
    GPU_HOST inline void AllReduce(const T* sendbuf, T* recvbuf, size_t count, Communicator<MPIBackend>* comm) {
        if (is_grouped == 0) {
            if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
                GPU_RT_CALL(UncGpuStreamSynchronize(stream));
            }
            MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Allreduce)(internal::mpi::buf_or_inplace(sendbuf), recvbuf, count,
                                                             internal::mpi::TypeMap<T>(),
                                                             internal::mpi::ReductOp2MPI_Op<OP>(), comm->mpi_comm));
        } else {
            MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Iallreduce)(
                internal::mpi::buf_or_inplace(sendbuf), recvbuf, count, internal::mpi::TypeMap<T>(),
                internal::mpi::ReductOp2MPI_Op<OP>(), comm->mpi_comm, &internal_reqs.emplace_back(MPI_REQUEST_NULL)));
        }
    }
    template <ReductionOperator OP, typename T>
    GPU_HOST inline void AllReduce(T* buffer, size_t count, Communicator<MPIBackend>* comm) {
        AllReduce<OP>(internal::IN_PLACE<T>, buffer, count, comm);
    }

    template <ReductionOperator OP, typename T>
    GPU_HOST inline void Reduce(const T* sendbuf, T* recvbuf, size_t count, int root, Communicator<MPIBackend>* comm) {
        if (sendbuf == internal::IN_PLACE<T> && comm->GlobalRank() != root) {
            sendbuf = recvbuf;
        }
        if (is_grouped == 0) {
            if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
                GPU_RT_CALL(UncGpuStreamSynchronize(stream));
            }
            MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Reduce)(internal::mpi::buf_or_inplace(sendbuf), recvbuf, count,
                                                          internal::mpi::TypeMap<T>(),
                                                          internal::mpi::ReductOp2MPI_Op<OP>(), root, comm->mpi_comm));
        } else {
            MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Ireduce)(internal::mpi::buf_or_inplace(sendbuf), recvbuf, count,
                                                           internal::mpi::TypeMap<T>(),
                                                           internal::mpi::ReductOp2MPI_Op<OP>(), root, comm->mpi_comm,
                                                           &internal_reqs.emplace_back(MPI_REQUEST_NULL)));
        }
    }
    template <ReductionOperator OP, typename T>
    GPU_HOST inline void Reduce(T* buffer, size_t count, int root, Communicator<MPIBackend>* comm) {
        Reduce<OP>(internal::IN_PLACE<T>, buffer, count, root, comm);
    }

    template <typename T>
    GPU_HOST inline void AlltoAll(const T* sendbuf, T* recvbuf, size_t count, Communicator<MPIBackend>* comm) {
        if (is_grouped == 0) {
            if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
                GPU_RT_CALL(UncGpuStreamSynchronize(stream));
            }
            MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Alltoall)(sendbuf, count, internal::mpi::TypeMap<T>(), recvbuf,
                                                            int(count), internal::mpi::TypeMap<T>(), comm->mpi_comm));
        } else {
            MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Ialltoall)(sendbuf, count, internal::mpi::TypeMap<T>(), recvbuf,
                                                             int(count), internal::mpi::TypeMap<T>(), comm->mpi_comm,
                                                             &internal_reqs.emplace_back(MPI_REQUEST_NULL)));
        }
    }
    template <typename T>
    GPU_HOST inline void AlltoAllv(const T* sendbuf, size_t* send_counts, size_t* send_displs, T* recvbuf,
                                   size_t* recv_counts, size_t* recv_displs, Communicator<MPIBackend>* comm) {
        {
            if (temp_send_counts.empty() && temp_send_displs.empty() && temp_recv_counts.empty() &&
                temp_recv_displs.empty()) {
                for (size_t i = 0; i < comm->GlobalSize(); i++) {
                    temp_send_counts.emplace_back(send_counts[i]);
                    temp_send_displs.emplace_back(send_displs[i]);
                    temp_recv_counts.emplace_back(recv_counts[i]);
                    temp_recv_displs.emplace_back(recv_displs[i]);
                }
                if (is_grouped == 0) {
                    if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
                        GPU_RT_CALL(UncGpuStreamSynchronize(stream));
                    }
                    MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Alltoallv)(
                        sendbuf, temp_send_counts.data(), temp_send_displs.data(), internal::mpi::TypeMap<T>(), recvbuf,
                        temp_recv_counts.data(), temp_recv_displs.data(), internal::mpi::TypeMap<T>(), comm->mpi_comm));
                    temp_send_counts.clear();
                    temp_send_displs.clear();
                    temp_recv_counts.clear();
                    temp_recv_displs.clear();
                } else {
                    MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Ialltoallv)(
                        sendbuf, temp_send_counts.data(), temp_send_displs.data(), internal::mpi::TypeMap<T>(), recvbuf,
                        temp_recv_counts.data(), temp_recv_displs.data(), internal::mpi::TypeMap<T>(), comm->mpi_comm,
                        &internal_reqs.emplace_back(MPI_REQUEST_NULL)));
                }
            }
        }
    }

    template <typename T>
    GPU_HOST inline void Broadcast(T* buffer, size_t count, int root, Communicator<MPIBackend>* comm) {
        if (is_grouped == 0) {
            if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
                GPU_RT_CALL(UncGpuStreamSynchronize(stream));
            }
            MPI_CALL(
                UNC_MPI_LARGE_COUNT_CALL(MPI_Bcast)(buffer, count, internal::mpi::TypeMap<T>(), root, comm->mpi_comm));
        } else {
            MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Ibcast)(buffer, count, internal::mpi::TypeMap<T>(), root,
                                                          comm->mpi_comm,
                                                          &internal_reqs.emplace_back(MPI_REQUEST_NULL)));
        }
    }

    template <typename T>
    GPU_HOST inline void Gather(const T* sendbuf, T* recvbuf, size_t count, int root, Communicator<MPIBackend>* comm) {
        if (is_grouped == 0) {
            if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
                GPU_RT_CALL(UncGpuStreamSynchronize(stream));
            }
            MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Gather)(internal::mpi::buf_or_inplace(sendbuf), count,
                                                          internal::mpi::TypeMap<T>(), recvbuf, count,
                                                          internal::mpi::TypeMap<T>(), root, comm->mpi_comm));
        } else {
            MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Igather)(
                internal::mpi::buf_or_inplace(sendbuf), count, internal::mpi::TypeMap<T>(), recvbuf, count,
                internal::mpi::TypeMap<T>(), root, comm->mpi_comm, &internal_reqs.emplace_back(MPI_REQUEST_NULL)));
        }
    }
    template <typename T>
    GPU_HOST inline void Gather(T* buffer, size_t count, int root, Communicator<MPIBackend>* comm) {
        Gather(internal::IN_PLACE<T>, buffer, count, root, comm);
    }

    template <typename T>
    GPU_HOST inline void Gatherv(const T* sendbuf, T* recvbuf, size_t* counts, size_t* displs, int root,
                                 Communicator<MPIBackend>* comm) {
        if (sendbuf == internal::IN_PLACE<T> && comm->GlobalRank() != root) {
            sendbuf = recvbuf;
        }
        if (temp_recv_counts.empty() && temp_recv_displs.empty()) {
            for (size_t i = 0; i < comm->GlobalSize(); i++) {
                temp_recv_counts.emplace_back(counts[i]);
                temp_recv_displs.emplace_back(displs[i]);
            }
            if (is_grouped == 0) {
                if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
                    GPU_RT_CALL(UncGpuStreamSynchronize(stream));
                }
                MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Gatherv)(
                    internal::mpi::buf_or_inplace(sendbuf), counts[comm->GlobalRank()], internal::mpi::TypeMap<T>(),
                    recvbuf, temp_recv_counts.data(), temp_recv_displs.data(), internal::mpi::TypeMap<T>(), root,
                    comm->mpi_comm));
                temp_recv_counts.clear();
                temp_recv_displs.clear();
            } else {
                MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Igatherv)(
                    internal::mpi::buf_or_inplace(sendbuf), counts[comm->GlobalRank()], internal::mpi::TypeMap<T>(),
                    recvbuf, temp_recv_counts.data(), temp_recv_displs.data(), internal::mpi::TypeMap<T>(), root,
                    comm->mpi_comm, &internal_reqs.emplace_back(MPI_REQUEST_NULL)));
            }
        }
    }
    template <typename T>
    GPU_HOST inline void Gatherv(T* buffer, size_t* counts, size_t* displs, int root, Communicator<MPIBackend>* comm) {
        Gatherv(internal::IN_PLACE<T>, buffer, counts, displs, root, comm);
    }

    template <typename T>
    GPU_HOST inline void Scatter(const T* sendbuf, T* recvbuf, size_t count, int root, Communicator<MPIBackend>* comm) {
        if (sendbuf == internal::IN_PLACE<T> && comm->GlobalRank() == root) {
            sendbuf = recvbuf;
            recvbuf = internal::IN_PLACE<T>;
        }
        if (is_grouped == 0) {
            if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
                GPU_RT_CALL(UncGpuStreamSynchronize(stream));
            }
            MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Scatter)(sendbuf, count, internal::mpi::TypeMap<T>(),
                                                           internal::mpi::buf_or_inplace(recvbuf), count,
                                                           internal::mpi::TypeMap<T>(), root, comm->mpi_comm));
        } else {
            MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Iscatter)(
                sendbuf, count, internal::mpi::TypeMap<T>(), internal::mpi::buf_or_inplace(recvbuf), count,
                internal::mpi::TypeMap<T>(), root, comm->mpi_comm, &internal_reqs.emplace_back(MPI_REQUEST_NULL)));
        }
    }
    template <typename T>
    GPU_HOST inline void Scatter(T* buffer, size_t count, int root, Communicator<MPIBackend>* comm) {
        Scatter(internal::IN_PLACE<T>, buffer, count, root, comm);
    }

    template <typename T>
    GPU_HOST inline void Scatterv(const T* sendbuf, T* recvbuf, size_t* counts, size_t* displs, int root,
                                  Communicator<MPIBackend>* comm) {
        if (sendbuf == internal::IN_PLACE<T> && comm->GlobalRank() == root) {
            sendbuf = recvbuf;
            recvbuf = internal::IN_PLACE<T>;
        }
        if (temp_send_counts.empty() && temp_send_displs.empty()) {
            for (size_t i = 0; i < comm->GlobalSize(); i++) {
                temp_send_counts.emplace_back(counts[i]);
                temp_send_displs.emplace_back(displs[i]);
            }
            if (is_grouped == 0) {
                if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
                    GPU_RT_CALL(UncGpuStreamSynchronize(stream));
                }
                MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Scatterv)(
                    sendbuf, temp_send_counts.data(), temp_send_displs.data(), internal::mpi::TypeMap<T>(),
                    internal::mpi::buf_or_inplace(recvbuf), counts[comm->GlobalRank()], internal::mpi::TypeMap<T>(),
                    root, comm->mpi_comm));
                temp_send_counts.clear();
                temp_send_displs.clear();
            } else {
                MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Iscatterv)(
                    sendbuf, temp_send_counts.data(), temp_send_displs.data(), internal::mpi::TypeMap<T>(),
                    internal::mpi::buf_or_inplace(recvbuf), counts[comm->GlobalRank()], internal::mpi::TypeMap<T>(),
                    root, comm->mpi_comm, &internal_reqs.emplace_back(MPI_REQUEST_NULL)));
            }
        }
    }

    template <typename T>
    GPU_HOST inline void Scatterv(T* buffer, size_t* counts, size_t* displs, int root, Communicator<MPIBackend>* comm) {
        Scatterv(internal::IN_PLACE<T>, buffer, counts, displs, root, comm);
    }

    GPU_HOST inline void WaitComm(){}

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Post(T* src_buffer, T* dest_buffer, size_t buffer_size, uint64_t* signal_location,
                                       uint64_t signal_val, int dest_process_id, Communicator<MPIBackend>* comm){}

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Acknowledge(T* dest_buffer, size_t buffer_size, uint64_t* signal_location,
                                              uint64_t signal_val, int src_process_id, Communicator<MPIBackend>* comm){}

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void AllGather(T* sendbuf, T* recvbuf, size_t count, Communicator<MPIBackend>* comm){}
    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void AllGather(T* buffer, size_t count, Communicator<MPIBackend>* comm){}

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void AllGatherv(T* sendbuf, T* recvbuf, size_t* counts, size_t* displs,
                                             Communicator<MPIBackend>* comm){}
    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void AllGatherv(T* buffer, size_t* counts, size_t* displs, Communicator<MPIBackend>* comm){}

    template <ThreadGroup SCOPE, ReductionOperator OP, typename T>
    GPU_DEVICE static inline void AllReduce(const T* sendbuf, T* recvbuf, size_t count, Communicator<MPIBackend>* comm){}
    template <ThreadGroup SCOPE, ReductionOperator OP, typename T>
    GPU_DEVICE static inline void AllReduce(T* buffer, size_t count, Communicator<MPIBackend>* comm){}

    template <ThreadGroup SCOPE, ReductionOperator OP, typename T>
    GPU_DEVICE static inline void Reduce(const T* sendbuf, T* recvbuf, size_t count, int root,
                                         Communicator<MPIBackend>* comm){}
    template <ThreadGroup SCOPE, ReductionOperator OP, typename T>
    GPU_DEVICE static inline void Reduce(T* buffer, size_t count, int root, Communicator<MPIBackend>* comm){}

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void AlltoAll(const T* sendbuf, T* recvbuf, size_t count, Communicator<MPIBackend>* comm){}
    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void AlltoAllv(const T* sendbuf, size_t* send_counts, size_t* send_displs, T* recvbuf,
                                            size_t* recv_counts, size_t* recv_displs, Communicator<MPIBackend>* comm){}

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Broadcast(T* buffer, size_t count, int root, Communicator<MPIBackend>* comm){}

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Gather(const T* sendbuf, T* recvbuf, size_t count, int root,
                                         Communicator<MPIBackend>* comm){}
    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Gather(T* buffer, size_t count, int root, Communicator<MPIBackend>* comm){}

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Gatherv(const T* sendbuf, T* recvbuf, size_t* counts, size_t* displs, int root,
                                          Communicator<MPIBackend>* comm){}
    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Gatherv(T* buffer, size_t* counts, size_t* displs, int root,
                                          Communicator<MPIBackend>* comm){}

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Scatter(const T* sendbuf, T* recvbuf, size_t count, int root,
                                          Communicator<MPIBackend>* comm){}
    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Scatter(T* buffer, size_t count, int root, Communicator<MPIBackend>* comm){}

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Scatterv(const T* sendbuf, T* recvbuf, size_t* counts, size_t* displs, int root,
                                           Communicator<MPIBackend>* comm){}
    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Scatterv(T* buffer, size_t* counts, size_t* displs, int root,
                                           Communicator<MPIBackend>* comm){}

    template <ThreadGroup SCOPE>
    GPU_DEVICE static inline void Wait(){}

    GPU_HOST ~Coordinator();
};
}  // namespace uniconn
#endif  // __UNICONN_INCLUDE_UNICONN_GPUMPI_COORDINATOR_HPP_
