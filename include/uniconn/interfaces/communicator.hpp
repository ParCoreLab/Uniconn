#ifndef __UNICONN_INCLUDE_UNICONN_INTERFACES_COMMUNICATOR_HPP_
#define __UNICONN_INCLUDE_UNICONN_INTERFACES_COMMUNICATOR_HPP_
#include "uniconn/utils.hpp"
namespace uniconn {
template <typename B = DefaultBackend>
struct Communicator {
    GPU_HOST Communicator();
    GPU_HOST inline Communicator<B>* toDevice();
    GPU_UNIFIED inline int GlobalSize();
    GPU_UNIFIED inline int GlobalRank();
    template <ThreadGroup SCOPE>
    GPU_DEVICE inline void Barrier();
    GPU_HOST inline void Barrier(UncGpuStream_t stream);
    GPU_HOST ~Communicator();
};

}  // namespace uniconn
#endif  // __UNICONN_INCLUDE_UNICONN_INTERFACES_COMMUNICATOR_HPP_
