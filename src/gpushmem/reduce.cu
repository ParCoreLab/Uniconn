#include "uniconn/gpushmem/coordinator.hpp"
#include "common.hpp"
// template <unsigned int blockSize>
// GPU_DEVICE void warpReduce(volatile int* sdata, unsigned int tid) {
//     if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
//     if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
//     if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
//     if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
//     if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
//     if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
// }
// template <unsigned int blockSize>
// GPU_KERNEL void reduce6(int* g_idata, int* g_odata, unsigned int n) {
//     extern __shared__ int sdata[];
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x * (blockSize * 2) + tid;
//     unsigned int gridSize = blockSize * 2 * gridDim.x;
//     sdata[tid] = 0;
//     while (i < n) {
//         sdata[tid] += g_idata[i] + g_idata[i + blockSize];
//         i += gridSize;
//     }
//     __syncthreads();
//     if (blockSize >= 512) {
//         if (tid < 256) {
//             sdata[tid] += sdata[tid + 256];
//         }
//         __syncthreads();
//     }
//     if (blockSize >= 256) {
//         if (tid < 128) {
//             sdata[tid] += sdata[tid + 128];
//         }
//         __syncthreads();
//     }
//     if (blockSize >= 128) {
//         if (tid < 64) {
//             sdata[tid] += sdata[tid + 64];
//         }
//         __syncthreads();
//     }
//     if (tid < 32) warpReduce(sdata, tid);
//     if (tid == 0) g_odata[blockIdx.x] = sdata[0];
// }

namespace uniconn {

#define DECL_FUNC(TYPENAME, TYPE, OPNAME, OP, MODE)                                                           \
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

UNC_SHMEM_REPT_REDUCE_TYPE_OP_MODE_HOST(DECL_FUNC)
#undef DECL_FUNC

#define DECL_FUNC(TYPENAME, TYPE, OPNAME, OP, MODE, SCOPE)                                                             \
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

UNC_SHMEM_REPT_REDUCE_TYPE_OP_MODE_GROUP_DEVICE(DECL_FUNC)
#undef DECL_FUNC

}  // namespace uniconn
