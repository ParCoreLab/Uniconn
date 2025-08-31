#include "uniconn/interfaces/kernelinfo.hpp"
#include "utils.hpp"
namespace uniconn {

GPU_HOST KernelInfo::KernelInfo()
    : kernel_func(nullptr), kernel_args(nullptr), grid_dims(1, 1, 1), block_dims(1, 1, 1), shared_mem(0) {}

GPU_HOST KernelInfo::KernelInfo(void* kernel_func, void** kernel_args, dim3 grid_dims, dim3 block_dims,
                                size_t shared_mem)
    : kernel_func(kernel_func),
      kernel_args(kernel_args),
      grid_dims(grid_dims),
      block_dims(block_dims),
      shared_mem(shared_mem) {}

// GPU_HOST void KernelInfo::launch(UncGpuStream_t stream) {
//     if (kernel_func != nullptr) {
//         GPU_RT_CALL(UncGpuLaunchKernel(kernel_func, grid_dims, block_dims, kernel_args, shared_mem, stream));
//     }
// }
// GPU_HOST void KernelInfo::launchCoop(UncGpuStream_t stream) {
//     if (kernel_func != nullptr) {
//         GPU_RT_CALL(UncGpuLaunchCooperativeKernel(kernel_func, grid_dims, block_dims, kernel_args, shared_mem, stream));
//     }
// }
}  // namespace uniconn