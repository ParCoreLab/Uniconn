#ifndef __UNICONN_SRC_GPUSHMEM_REDUCE_BASE_HPP_
#define __UNICONN_SRC_GPUSHMEM_REDUCE_BASE_HPP_



#include "common.hpp"
#include "uniconn/gpushmem/coordinator.hpp"
namespace uniconn {

#define DECL_FUNC_HOST(TYPENAME, TYPE, OPNAME, OP, MODE)                                                      \
    template <>                                                                                               \
    template <>                                                                                               \
    GPU_HOST void Coordinator<GpushmemBackend, MODE>::Reduce<OP, TYPE>(                                       \
        const TYPE* sendbuf, TYPE* recvbuf, size_t count, int root, Communicator<GpushmemBackend>* comm) {    \
        if constexpr (MODE == LaunchMode::HostDriven || MODE == LaunchMode::LimitedDevice) {                  \
        }                                                                                                     \
    }                                                                                                         \
                                                                                                              \
    template <>                                                                                               \
    template <>                                                                                               \
    GPU_HOST void Coordinator<GpushmemBackend, MODE>::Reduce<OP, TYPE>(TYPE * buffer, size_t count, int root, \
                                                                       Communicator<GpushmemBackend>* comm) { \
        Reduce<OP>(buffer, buffer, count, root, comm);                                                        \
    }

#define DECL_FUNC_DEVICE(TYPENAME, TYPE, OPNAME, OP, MODE, SCOPE)                                                      \
    template <>                                                                                                        \
    template <>                                                                                                        \
    GPU_DEVICE void Coordinator<GpushmemBackend, MODE>::Reduce<SCOPE, OP, TYPE>(                                       \
        const TYPE* sendbuf, TYPE* recvbuf, size_t count, int root, Communicator<GpushmemBackend>* comm) {             \
        if constexpr (MODE == LaunchMode::FullDevice) {                                                                \
        }                                                                                                              \
    }                                                                                                                  \
                                                                                                                       \
    template <>                                                                                                        \
    template <>                                                                                                        \
    GPU_DEVICE void Coordinator<GpushmemBackend, MODE>::Reduce<SCOPE, OP, TYPE>(TYPE * buffer, size_t count, int root, \
                                                                                Communicator<GpushmemBackend>* comm) { \
        Reduce<SCOPE, OP>(buffer, buffer, count, root, comm);                                                          \
    }
}  // namespace uniconn
#endif // __UNICONN_SRC_GPUSHMEM_REDUCE_BASE_HPP_