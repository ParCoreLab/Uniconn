#ifndef __UNICONN_INCLUDE_UNICONN_INTERFACES_KERNELINFO_HPP_
#define __UNICONN_INCLUDE_UNICONN_INTERFACES_KERNELINFO_HPP_
#include "uniconn/utils.hpp"
namespace uniconn {
struct KernelInfo {
    void* kernel_func;
    void** kernel_args;
    dim3 grid_dims;
    dim3 block_dims;
    size_t shared_mem;
    GPU_HOST KernelInfo();

    GPU_HOST KernelInfo(void* kernel_func, void** kernel_args, dim3 grid_dims, dim3 block_dims,
                        size_t shared_mem);

    GPU_HOST void __forceinline__ launch(UncGpuStream_t stream);
    GPU_HOST void __forceinline__ launchCoop(UncGpuStream_t stream);
};

GPU_HOST __forceinline__ void KernelInfo::launch(UncGpuStream_t stream) {
    if (kernel_func != nullptr) {
        GPU_RT_CALL(UncGpuLaunchKernel(kernel_func, grid_dims, block_dims, kernel_args, shared_mem, stream));
    }
}
GPU_HOST __forceinline__ void KernelInfo::launchCoop(UncGpuStream_t stream) {
    if (kernel_func != nullptr) {
        GPU_RT_CALL(UncGpuLaunchCooperativeKernel(kernel_func, grid_dims, block_dims, kernel_args, shared_mem, stream));
    }
}

}  // namespace uniconn
#endif  // __UNICONN_INCLUDE_UNICONN_INTERFACES_KERNELINFO_HPP_