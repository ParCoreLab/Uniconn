#include "common.hpp"
#include "uniconn/gpushmem/coordinator.hpp"
namespace uniconn {

template <LaunchMode MODE>
GPU_HOST Coordinator<GpushmemBackend, MODE>::Coordinator(UncGpuStream_t stream) : stream(stream), kernel_info() {}

template GPU_HOST Coordinator<GpushmemBackend, LaunchMode::HostDriven>::Coordinator(UncGpuStream_t stream);
template GPU_HOST Coordinator<GpushmemBackend, LaunchMode::LimitedDevice>::Coordinator(UncGpuStream_t stream);
template GPU_HOST Coordinator<GpushmemBackend, LaunchMode::FullDevice>::Coordinator(UncGpuStream_t stream);

template <LaunchMode M>
template <LaunchMode MODE>
GPU_HOST void Coordinator<GpushmemBackend, M>::bindKernel(void* kernel_func, dim3 grid_dims, dim3 block_dims,
                                                          size_t shared_mem, void** kernel_args) {
    if constexpr (MODE == M) {
        kernel_info.kernel_func = kernel_func;
        kernel_info.kernel_args = kernel_args;
        kernel_info.grid_dims = grid_dims;
        kernel_info.block_dims = block_dims;
        kernel_info.shared_mem = shared_mem;
    }
}

template GPU_HOST void Coordinator<GpushmemBackend, LaunchMode::HostDriven>::bindKernel<LaunchMode::HostDriven>(
    void* kernel_func, dim3 grid_dims, dim3 block_dims, size_t shared_mem, void** kernel_args);
template GPU_HOST void Coordinator<GpushmemBackend, LaunchMode::HostDriven>::bindKernel<LaunchMode::LimitedDevice>(
    void* kernel_func, dim3 grid_dims, dim3 block_dims, size_t shared_mem, void** kernel_args);
template GPU_HOST void Coordinator<GpushmemBackend, LaunchMode::HostDriven>::bindKernel<LaunchMode::FullDevice>(
    void* kernel_func, dim3 grid_dims, dim3 block_dims, size_t shared_mem, void** kernel_args);

template GPU_HOST void Coordinator<GpushmemBackend, LaunchMode::LimitedDevice>::bindKernel<LaunchMode::HostDriven>(
    void* kernel_func, dim3 grid_dims, dim3 block_dims, size_t shared_mem, void** kernel_args);
template GPU_HOST void Coordinator<GpushmemBackend, LaunchMode::LimitedDevice>::bindKernel<LaunchMode::LimitedDevice>(
    void* kernel_func, dim3 grid_dims, dim3 block_dims, size_t shared_mem, void** kernel_args);
template GPU_HOST void Coordinator<GpushmemBackend, LaunchMode::LimitedDevice>::bindKernel<LaunchMode::FullDevice>(
    void* kernel_func, dim3 grid_dims, dim3 block_dims, size_t shared_mem, void** kernel_args);

template GPU_HOST void Coordinator<GpushmemBackend, LaunchMode::FullDevice>::bindKernel<LaunchMode::HostDriven>(
    void* kernel_func, dim3 grid_dims, dim3 block_dims, size_t shared_mem, void** kernel_args);
template GPU_HOST void Coordinator<GpushmemBackend, LaunchMode::FullDevice>::bindKernel<LaunchMode::LimitedDevice>(
    void* kernel_func, dim3 grid_dims, dim3 block_dims, size_t shared_mem, void** kernel_args);
template GPU_HOST void Coordinator<GpushmemBackend, LaunchMode::FullDevice>::bindKernel<LaunchMode::FullDevice>(
    void* kernel_func, dim3 grid_dims, dim3 block_dims, size_t shared_mem, void** kernel_args);

// template <LaunchMode MODE>
// GPU_HOST void Coordinator<GpushmemBackend, MODE>::LaunchKernel() {
//     if constexpr (MODE == LaunchMode::HostDriven || MODE == LaunchMode::LimitedDevice) {
//         kernel_info.launch(stream);

//     } else if constexpr (MODE == LaunchMode::FullDevice) {
//         GPU_RT_CALL((UncGpuError_t)nvshmemx_collective_launch(kernel_info.kernel_func, kernel_info.grid_dims,
//                                                               kernel_info.block_dims, kernel_info.kernel_args,
//                                                               kernel_info.shared_mem, stream));
//     }
// }

// template GPU_HOST void Coordinator<GpushmemBackend, LaunchMode::HostDriven>::LaunchKernel();
// template GPU_HOST void Coordinator<GpushmemBackend, LaunchMode::LimitedDevice>::LaunchKernel();
// template GPU_HOST void Coordinator<GpushmemBackend, LaunchMode::FullDevice>::LaunchKernel();

// template <LaunchMode MODE>
// GPU_HOST void Coordinator<GpushmemBackend, MODE>::CommStart() {}

// template GPU_HOST void Coordinator<GpushmemBackend, LaunchMode::HostDriven>::CommStart();
// template GPU_HOST void Coordinator<GpushmemBackend, LaunchMode::LimitedDevice>::CommStart();
// template GPU_HOST void Coordinator<GpushmemBackend, LaunchMode::FullDevice>::CommStart();

// template <LaunchMode MODE>
// GPU_HOST void Coordinator<GpushmemBackend, MODE>::CommEnd() {
//     // if constexpr (MODE == LaunchMode::HostDriven || MODE == LaunchMode::LimitedDevice) {
//     //     nvshmemx_quiet_on_stream(this->stream);
//     // }
// }

// template GPU_HOST void Coordinator<GpushmemBackend, LaunchMode::HostDriven>::CommEnd();
// template GPU_HOST void Coordinator<GpushmemBackend, LaunchMode::LimitedDevice>::CommEnd();
// template GPU_HOST void Coordinator<GpushmemBackend, LaunchMode::FullDevice>::CommEnd();

// template <LaunchMode MODE>
// GPU_HOST void Coordinator<GpushmemBackend, MODE>::WaitComm() {}
// template GPU_HOST void Coordinator<GpushmemBackend, LaunchMode::HostDriven>::WaitComm();
// template GPU_HOST void Coordinator<GpushmemBackend, LaunchMode::LimitedDevice>::WaitComm();
// template GPU_HOST void Coordinator<GpushmemBackend, LaunchMode::FullDevice>::WaitComm();

// template <LaunchMode MODE>
// template <ThreadGroup SCOPE>
// GPU_DEVICE void Coordinator<GpushmemBackend, MODE>::Wait() {
//     if constexpr (MODE == LaunchMode::FullDevice) {
//         if constexpr (SCOPE == ThreadGroup::BLOCK) {
//             cg::thread_block cta = cg::this_thread_block();
//             cta.sync();
//             if (!cta.thread_rank()) {
//                 nvshmem_quiet();
//             }
//         } else if constexpr (SCOPE == ThreadGroup::WARP) {
//             cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
//             warp.sync();
//             if (!warp.thread_rank()) {
//                 nvshmem_quiet();
//             }
//         } else if constexpr (SCOPE == ThreadGroup::THREAD) {
//             nvshmem_quiet();
//         }
//     }
// }

// template GPU_DEVICE void Coordinator<GpushmemBackend, LaunchMode::HostDriven>::Wait<ThreadGroup::BLOCK>();
// template GPU_DEVICE void Coordinator<GpushmemBackend, LaunchMode::HostDriven>::Wait<ThreadGroup::WARP>();
// template GPU_DEVICE void Coordinator<GpushmemBackend, LaunchMode::HostDriven>::Wait<ThreadGroup::THREAD>();

// template GPU_DEVICE void Coordinator<GpushmemBackend, LaunchMode::LimitedDevice>::Wait<ThreadGroup::BLOCK>();
// template GPU_DEVICE void Coordinator<GpushmemBackend, LaunchMode::LimitedDevice>::Wait<ThreadGroup::WARP>();
// template GPU_DEVICE void Coordinator<GpushmemBackend, LaunchMode::LimitedDevice>::Wait<ThreadGroup::THREAD>();

// template GPU_DEVICE void Coordinator<GpushmemBackend, LaunchMode::FullDevice>::Wait<ThreadGroup::BLOCK>();
// template GPU_DEVICE void Coordinator<GpushmemBackend, LaunchMode::FullDevice>::Wait<ThreadGroup::WARP>();
// template GPU_DEVICE void Coordinator<GpushmemBackend, LaunchMode::FullDevice>::Wait<ThreadGroup::THREAD>();

template <LaunchMode MODE>
GPU_HOST Coordinator<GpushmemBackend, MODE>::~Coordinator() {}

template GPU_HOST Coordinator<GpushmemBackend, LaunchMode::HostDriven>::~Coordinator();
template GPU_HOST Coordinator<GpushmemBackend, LaunchMode::LimitedDevice>::~Coordinator();
template GPU_HOST Coordinator<GpushmemBackend, LaunchMode::FullDevice>::~Coordinator();
}  // namespace uniconn
