#include "uniconn/gpuccl/coordinator.hpp"
#include "common.hpp"
namespace uniconn {

GPU_HOST Coordinator<GpucclBackend, LaunchMode::HostDriven>::Coordinator(UncGpuStream_t stream)
    : stream(stream), kernel_info() {}

template <LaunchMode MODE>
GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::bindKernel(void* kernel_func, dim3 grid_dims,
                                                                             dim3 block_dims, size_t shared_mem,
                                                                             void** kernel_args) {
    if constexpr (MODE == LaunchMode::HostDriven) {
        kernel_info.kernel_func = kernel_func;
        kernel_info.kernel_args = kernel_args;
        kernel_info.grid_dims = grid_dims;
        kernel_info.block_dims = block_dims;
        kernel_info.shared_mem = shared_mem;
    }
}
template GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::bindKernel<LaunchMode::HostDriven>(
    void* kernel_func, dim3 grid_dims, dim3 block_dims, size_t shared_mem, void** kernel_args);
template GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::bindKernel<LaunchMode::LimitedDevice>(
    void* kernel_func, dim3 grid_dims, dim3 block_dims, size_t shared_mem, void** kernel_args);
template GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::bindKernel<LaunchMode::FullDevice>(
    void* kernel_func, dim3 grid_dims, dim3 block_dims, size_t shared_mem, void** kernel_args);
    
// GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::LaunchKernel() { kernel_info.launch(stream); }

// GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::CommStart() { NCCL_CALL(ncclGroupStart()); }

// GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::CommEnd() {
//     NCCL_CALL(ncclGroupEnd());
// }

// GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::WaitComm() {}

// template <ThreadGroup SCOPE>
// GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Wait() {}

GPU_HOST Coordinator<GpucclBackend, LaunchMode::HostDriven>::~Coordinator() {}



// template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Wait<ThreadGroup::BLOCK>();
// template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Wait<ThreadGroup::WARP>();
// template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Wait<ThreadGroup::THREAD>();
}  // namespace uniconn
