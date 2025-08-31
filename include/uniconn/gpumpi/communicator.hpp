#ifndef __UNICONN_INCLUDE_UNICONN_GPUMPI_COMMUNICATOR_HPP_
#define __UNICONN_INCLUDE_UNICONN_GPUMPI_COMMUNICATOR_HPP_

#include "common.hpp"
#include "uniconn/interfaces/communicator.hpp"
namespace uniconn {

template <>
struct Communicator<MPIBackend> {
    MPI_Comm mpi_comm;
    GPU_HOST Communicator() ;
    GPU_HOST inline Communicator<MPIBackend>* toDevice() { return nullptr; }
    GPU_UNIFIED inline int GlobalSize() {
        int ret = 0;
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 0
        MPI_CALL(MPI_Comm_size(mpi_comm, &ret));
#endif
        return ret;
    }
    GPU_UNIFIED inline int GlobalRank() {
        int ret = -1;
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 0
        MPI_CALL(MPI_Comm_rank(mpi_comm, &ret));
#endif
        return ret;
    }
    template <ThreadGroup SCOPE>
    GPU_DEVICE inline void Barrier() {}
    GPU_HOST inline void Barrier(UncGpuStream_t stream) {
        if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
            GPU_RT_CALL(UncGpuStreamSynchronize(stream));
        }
        MPI_CALL(MPI_Barrier(mpi_comm));
    }
    GPU_HOST ~Communicator() ;
};

}  // namespace uniconn

#endif  // __UNICONN_INCLUDE_UNICONN_GPUMPI_COMMUNICATOR_HPP_
