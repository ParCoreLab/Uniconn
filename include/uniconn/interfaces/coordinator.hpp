#ifndef __UNICONN_INCLUDE_UNICONN_INTERFACES_COORDINATOR_HPP_
#define __UNICONN_INCLUDE_UNICONN_INTERFACES_COORDINATOR_HPP_

#include "communicator.hpp"
#include "kernelinfo.hpp"

namespace uniconn {

template <typename B = DefaultBackend, LaunchMode M = DefaultLaunchType>
class Coordinator {
   public:
    GPU_HOST Coordinator(UncGpuStream_t stream);

    template <LaunchMode MODE>
    GPU_HOST void bindKernel(void* kernel_func, dim3 grid_dims, dim3 block_dims, size_t shared_mem, void** kernel_args);

    GPU_HOST inline void LaunchKernel();

    GPU_HOST inline void CommStart();

    GPU_HOST inline void CommEnd();

    template <typename T>
    GPU_HOST inline void Post(T* src_buffer, T* dest_buffer, size_t buffer_size, uint64_t* signal_location,
                              uint64_t signal_val, int dest_process_id, Communicator<B>* comm);

    template <typename T>
    GPU_HOST inline void Acknowledge(T* dest_buffer, size_t buffer_size, uint64_t* signal_location, uint64_t signal_val,
                                     int src_process_id, Communicator<B>* comm);

    template <typename T>
    GPU_HOST inline void AllGather(T* sendbuf, T* recvbuf, size_t count, Communicator<B>* comm);
    template <typename T>
    GPU_HOST inline void AllGather(T* buffer, size_t count, Communicator<B>* comm);

    template <typename T>
    GPU_HOST inline void AllGatherv(T* sendbuf, T* recvbuf, size_t* counts, size_t* displs, Communicator<B>* comm);
    template <typename T>
    GPU_HOST inline void AllGatherv(T* buffer, size_t* counts, size_t* displs, Communicator<B>* comm);

    template <ReductionOperator OP, typename T>
    GPU_HOST inline void AllReduce(const T* sendbuf, T* recvbuf, size_t count, Communicator<B>* comm);
    template <ReductionOperator OP, typename T>
    GPU_HOST inline void AllReduce(T* buffer, size_t count, Communicator<B>* comm);

    template <ReductionOperator OP, typename T>
    GPU_HOST inline void Reduce(const T* sendbuf, T* recvbuf, size_t count, int root, Communicator<B>* comm);
    template <ReductionOperator OP, typename T>
    GPU_HOST inline void Reduce(T* buffer, size_t count, int root, Communicator<B>* comm);

    template <typename T>
    GPU_HOST inline void AlltoAll(const T* sendbuf, T* recvbuf, size_t count, Communicator<B>* comm);
    template <typename T>
    GPU_HOST inline void AlltoAllv(const T* sendbuf, size_t* send_counts, size_t* send_displs, T* recvbuf,
                                   size_t* recv_counts, size_t* recv_displs, Communicator<B>* comm);

    template <typename T>
    GPU_HOST inline void Broadcast(T* buffer, size_t count, int root, Communicator<B>* comm);

    template <typename T>
    GPU_HOST inline void Gather(const T* sendbuf, T* recvbuf, size_t count, int root, Communicator<B>* comm);
    template <typename T>
    GPU_HOST inline void Gather(T* buffer, size_t count, int root, Communicator<B>* comm);

    template <typename T>
    GPU_HOST inline void Gatherv(const T* sendbuf, T* recvbuf, size_t* counts, size_t* displs, int root,
                                 Communicator<B>* comm);
    template <typename T>
    GPU_HOST inline void Gatherv(T* buffer, size_t* counts, size_t* displs, int root, Communicator<B>* comm);

    template <typename T>
    GPU_HOST inline void Scatter(const T* sendbuf, T* recvbuf, size_t count, int root, Communicator<B>* comm);
    template <typename T>
    GPU_HOST inline void Scatter(T* buffer, size_t count, int root, Communicator<B>* comm);

    template <typename T>
    GPU_HOST inline void Scatterv(const T* sendbuf, T* recvbuf, size_t* counts, size_t* displs, int root,
                                  Communicator<B>* comm);

    template <typename T>
    GPU_HOST inline void Scatterv(T* buffer, size_t* counts, size_t* displs, int root, Communicator<B>* comm);

    GPU_HOST inline void WaitComm();

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Post(T* src_buffer, T* dest_buffer, size_t buffer_size, uint64_t* signal_location,
                                       uint64_t signal_val, int dest_process_id, Communicator<B>* comm);

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Acknowledge(T* dest_buffer, size_t buffer_size, uint64_t* signal_location,
                                              uint64_t signal_val, int src_process_id, Communicator<B>* comm);

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void AllGather(T* sendbuf, T* recvbuf, size_t count, Communicator<B>* comm);
    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void AllGather(T* buffer, size_t count, Communicator<B>* comm);

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void AllGatherv(T* sendbuf, T* recvbuf, size_t* counts, size_t* displs,
                                             Communicator<B>* comm);
    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void AllGatherv(T* buffer, size_t* counts, size_t* displs, Communicator<B>* comm);

    template <ThreadGroup SCOPE, ReductionOperator OP, typename T>
    GPU_DEVICE static inline void AllReduce(const T* sendbuf, T* recvbuf, size_t count, Communicator<B>* comm);
    template <ThreadGroup SCOPE, ReductionOperator OP, typename T>
    GPU_DEVICE static inline void AllReduce(T* buffer, size_t count, Communicator<B>* comm);

    template <ThreadGroup SCOPE, ReductionOperator OP, typename T>
    GPU_DEVICE static inline void Reduce(const T* sendbuf, T* recvbuf, size_t count, int root, Communicator<B>* comm);
    template <ThreadGroup SCOPE, ReductionOperator OP, typename T>
    GPU_DEVICE static inline void Reduce(T* buffer, size_t count, int root, Communicator<B>* comm);

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void AlltoAll(const T* sendbuf, T* recvbuf, size_t count, Communicator<B>* comm);
    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void AlltoAllv(const T* sendbuf, size_t* send_counts, size_t* send_displs, T* recvbuf,
                                            size_t* recv_counts, size_t* recv_displs, Communicator<B>* comm);

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Broadcast(T* buffer, size_t count, int root, Communicator<B>* comm);

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Gather(const T* sendbuf, T* recvbuf, size_t count, int root, Communicator<B>* comm);
    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Gather(T* buffer, size_t count, int root, Communicator<B>* comm);

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Gatherv(const T* sendbuf, T* recvbuf, size_t* counts, size_t* displs, int root,
                                          Communicator<B>* comm);
    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Gatherv(T* buffer, size_t* counts, size_t* displs, int root, Communicator<B>* comm);

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Scatter(const T* sendbuf, T* recvbuf, size_t count, int root, Communicator<B>* comm);
    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Scatter(T* buffer, size_t count, int root, Communicator<B>* comm);

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Scatterv(const T* sendbuf, T* recvbuf, size_t* counts, size_t* displs, int root,
                                           Communicator<B>* comm);

    template <ThreadGroup SCOPE, typename T>
    GPU_DEVICE static inline void Scatterv(T* buffer, size_t* counts, size_t* displs, int root, Communicator<B>* comm);
    template <ThreadGroup SCOPE>
    GPU_DEVICE static inline void Wait();
    GPU_HOST ~Coordinator();
};
}  // namespace uniconn

#endif  // __UNICONN_INCLUDE_UNICONN_INTERFACES_COORDINATOR_HPP_
