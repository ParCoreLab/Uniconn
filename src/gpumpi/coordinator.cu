#include "common.hpp"
#include "uniconn/gpumpi/coordinator.hpp"
namespace uniconn {

GPU_HOST Coordinator<MPIBackend, LaunchMode::HostDriven>::Coordinator(UncGpuStream_t stream)
    : stream(stream),
      internal_reqs(),
      temp_send_counts(),
      temp_send_displs(),
      temp_recv_counts(),
      temp_recv_displs(),
      kernel_info(),
      is_grouped(0) {}

template <LaunchMode MODE>
GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::bindKernel(void* kernel_func, dim3 grid_dims,
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
template GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::bindKernel<LaunchMode::HostDriven>(
    void* kernel_func, dim3 grid_dims, dim3 block_dims, size_t shared_mem, void** kernel_args);
template GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::bindKernel<LaunchMode::LimitedDevice>(
    void* kernel_func, dim3 grid_dims, dim3 block_dims, size_t shared_mem, void** kernel_args);
template GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::bindKernel<LaunchMode::FullDevice>(
    void* kernel_func, dim3 grid_dims, dim3 block_dims, size_t shared_mem, void** kernel_args);

// GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::LaunchKernel() { kernel_info.launch(stream); }

// GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::CommStart() {
//     if (is_grouped == 0) {
//         if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
//             GPU_RT_CALL(UncGpuStreamSynchronize(stream));
//         }
//         is_grouped++;
//     }
// }

// GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::CommEnd() {
//     if (is_grouped == 1) {
//         is_grouped--;
//         MPI_CALL(MPI_Waitall(internal_reqs.size(), internal_reqs.data(), MPI_STATUSES_IGNORE));
//         internal_reqs.clear();
//     }
// }

// GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::WaitComm() {
//     // MPI_CALL(MPI_Waitall(internal_reqs.size(), internal_reqs.data(), MPI_STATUSES_IGNORE));
//     // internal_reqs.clear();
// }

// template <ThreadGroup SCOPE>
// GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Wait() {}

GPU_HOST Coordinator<MPIBackend, LaunchMode::HostDriven>::~Coordinator() {}

// template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Wait<ThreadGroup::BLOCK>();
// template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Wait<ThreadGroup::WARP>();
// template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Wait<ThreadGroup::THREAD>();
}  // namespace uniconn
