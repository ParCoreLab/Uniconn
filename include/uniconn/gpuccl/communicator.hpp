#ifndef __UNICONN_INCLUDE_UNICONN_GPUCCL_COMMUNICATOR_HPP_
#define __UNICONN_INCLUDE_UNICONN_GPUCCL_COMMUNICATOR_HPP_
#include "common.hpp"
#include "memory.hpp"
#include "uniconn/interfaces/communicator.hpp"
namespace uniconn {
template <>
struct Communicator<GpucclBackend> {
    ncclComm_t nccl_comm;
    unsigned char* barrier_buf;

    GPU_HOST Communicator();
    GPU_HOST inline Communicator<GpucclBackend>* toDevice() { return nullptr; }
    GPU_UNIFIED inline int GlobalSize() {
        int ret = 0;
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 0
        NCCL_CALL(ncclCommCount(nccl_comm, &ret));
#endif
        return ret;
    }
    GPU_UNIFIED inline int GlobalRank() {
        int ret = -1;
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 0
        NCCL_CALL(ncclCommUserRank(nccl_comm, &ret));
#endif
        return ret;
    }
    template <ThreadGroup SCOPE>
    GPU_DEVICE inline void Barrier() {}
    GPU_HOST inline void Barrier(UncGpuStream_t stream) {
        NCCL_CALL(ncclAllReduce((const void*)barrier_buf, (void*)barrier_buf, 1,
                                internal::gpuccl::TypeMap<unsigned char>(), ncclSum, nccl_comm, stream));
    }
    GPU_HOST ~Communicator();
};

}  // namespace uniconn

#endif  // __UNICONN_INCLUDE_UNICONN_GPUCCL_COMMUNICATOR_HPP_
