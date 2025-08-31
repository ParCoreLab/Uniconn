#ifndef __UNICONN_SRC_GPUSHMEM_ALLREDUCE_BASE_HPP_
#define __UNICONN_SRC_GPUSHMEM_ALLREDUCE_BASE_HPP_

#include "common.hpp"
#include "uniconn/gpushmem/coordinator.hpp"
namespace uniconn {

#define DECL_FUNC_HOST(TYPENAME, TYPE, OPNAME, OP, MODE)                                                            \
    template <>                                                                                                  \
    template <>                                                                                                  \
    GPU_HOST void Coordinator<GpushmemBackend, MODE>::AllReduce<OP, TYPE>(                                       \
        const TYPE* sendbuf, TYPE* recvbuf, size_t count, Communicator<GpushmemBackend>* comm) {                 \
        if constexpr (MODE == LaunchMode::HostDriven || MODE == LaunchMode::LimitedDevice) {                     \
            nvshmemx_##TYPENAME##_##OPNAME##_reduce_on_stream(comm->nvshmem_comm, recvbuf, sendbuf, count,       \
                                                              this->stream);                                     \
        }                                                                                                        \
    }                                                                                                            \
                                                                                                                 \
    template <>                                                                                                  \
    template <>                                                                                                  \
    GPU_HOST void Coordinator<GpushmemBackend, MODE>::AllReduce<OP, TYPE>(TYPE * buffer, size_t count,           \
                                                                          Communicator<GpushmemBackend>* comm) { \
        AllReduce<OP>(buffer, buffer, count, comm);                                                              \
    }

#define DECL_FUNC_DEVICE(TYPENAME, TYPE, OPNAME, OP, MODE, SCOPE)                                                \
    template <>                                                                                             \
    template <>                                                                                             \
    GPU_DEVICE void Coordinator<GpushmemBackend, MODE>::AllReduce<SCOPE, OP, TYPE>(                         \
        const TYPE* sendbuf, TYPE* recvbuf, size_t count, Communicator<GpushmemBackend>* comm) {            \
        if constexpr (MODE == LaunchMode::FullDevice) {                                                     \
            if constexpr (SCOPE == ThreadGroup::BLOCK) {                                                    \
                nvshmemx_##TYPENAME##_##OPNAME##_reduce_block(comm->nvshmem_comm, recvbuf, sendbuf, count); \
            } else if constexpr (SCOPE == ThreadGroup::WARP) {                                              \
                nvshmemx_##TYPENAME##_##OPNAME##_reduce_warp(comm->nvshmem_comm, recvbuf, sendbuf, count);  \
            } else if constexpr (SCOPE == ThreadGroup::THREAD) {                                            \
                nvshmem_##TYPENAME##_##OPNAME##_reduce(comm->nvshmem_comm, recvbuf, sendbuf, count);        \
            }                                                                                               \
        }                                                                                                   \
    }                                                                                                       \
                                                                                                            \
    template <>                                                                                             \
    template <>                                                                                             \
    GPU_DEVICE void Coordinator<GpushmemBackend, MODE>::AllReduce<SCOPE, OP, TYPE>(                         \
        TYPE * buffer, size_t count, Communicator<GpushmemBackend>* comm) {                                 \
        AllReduce<SCOPE, OP>(buffer, buffer, count, comm);                                                  \
    }
}  // namespace uniconn
#endif  // __UNICONN_SRC_GPUSHMEM_ALLREDUCE_BASE_HPP_