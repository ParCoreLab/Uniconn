#include "uniconn/gpuccl/coordinator.hpp"
#include "common.hpp"
namespace uniconn {

// template <typename T>
// GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Post(T* src_buffer, T* dest_buffer,
//                                                                        size_t buffer_size, uint64_t* signal_location,
//                                                                        uint64_t signal_val, int dest_process_id,
//                                                                        Communicator<GpucclBackend>* comm) {
    
//     NCCL_CALL(ncclSend((const void*)src_buffer, buffer_size, internal::gpuccl::TypeMap<T>(), dest_process_id,
//                        comm->nccl_comm, stream));
// }

// template <typename T>
// GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Acknowledge(T* dest_buffer, size_t buffer_size,
//                                                                               uint64_t* signal_location,
//                                                                               uint64_t signal_val, int src_process_id,
//                                                                               Communicator<GpucclBackend>* comm) {
   
//     NCCL_CALL(ncclRecv((void*)dest_buffer, buffer_size, internal::gpuccl::TypeMap<T>(), src_process_id, comm->nccl_comm,
//                        stream));
// }
// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Post(T* src_buffer, T* dest_buffer,
//                                                                          size_t buffer_size, uint64_t* signal_location,
//                                                                          uint64_t signal_val, int dest_process_id,
//                                                                          Communicator<GpucclBackend>* comm) {}

// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Acknowledge(T* dest_buffer, size_t buffer_size,
//                                                                                 uint64_t* signal_location,
//                                                                                 uint64_t signal_val, int src_process_id,
//                                                                                 Communicator<GpucclBackend>* comm) {}
// #define DECL_FUNC(TYPE)                                                                                                \
//     template GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Post<TYPE>(                             \
//         TYPE * src_buffer, TYPE * dest_buffer, size_t buffer_size, uint64_t* signal_location, uint64_t signal_val,     \
//         int dest_process_id, Communicator<GpucclBackend>* comm);                                                       \
//     template GPU_HOST void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Acknowledge<TYPE>(                      \
//         TYPE * dest_buffer, size_t buffer_size, uint64_t* signal_location, uint64_t signal_val, int src_process_id,    \
//         Communicator<GpucclBackend>* comm);                                                                            \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Post<ThreadGroup::BLOCK, TYPE>(       \
//         TYPE * src_buffer, TYPE * dest_buffer, size_t buffer_size, uint64_t* signal_location, uint64_t signal_val,     \
//         int dest_process_id, Communicator<GpucclBackend>* comm);                                                       \
//     template GPU_DEVICE void                                                                                           \
//     Coordinator<GpucclBackend, LaunchMode::HostDriven>::Acknowledge<ThreadGroup::BLOCK, TYPE>(                         \
//         TYPE * dest_buffer, size_t buffer_size, uint64_t* signal_location, uint64_t signal_val, int src_process_id,    \
//         Communicator<GpucclBackend>* comm);                                                                            \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Post<ThreadGroup::WARP, TYPE>(        \
//         TYPE * src_buffer, TYPE * dest_buffer, size_t buffer_size, uint64_t* signal_location, uint64_t signal_val,     \
//         int dest_process_id, Communicator<GpucclBackend>* comm);                                                       \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Acknowledge<ThreadGroup::WARP, TYPE>( \
//         TYPE * dest_buffer, size_t buffer_size, uint64_t* signal_location, uint64_t signal_val, int src_process_id,    \
//         Communicator<GpucclBackend>* comm);                                                                            \
//     template GPU_DEVICE void Coordinator<GpucclBackend, LaunchMode::HostDriven>::Post<ThreadGroup::THREAD, TYPE>(      \
//         TYPE * src_buffer, TYPE * dest_buffer, size_t buffer_size, uint64_t* signal_location, uint64_t signal_val,     \
//         int dest_process_id, Communicator<GpucclBackend>* comm);                                                       \
//     template GPU_DEVICE void                                                                                           \
//     Coordinator<GpucclBackend, LaunchMode::HostDriven>::Acknowledge<ThreadGroup::THREAD, TYPE>(                        \
//         TYPE * dest_buffer, size_t buffer_size, uint64_t* signal_location, uint64_t signal_val, int src_process_id,    \
//         Communicator<GpucclBackend>* comm);

// UNC_GPUCCL_REPT(DECL_FUNC)

}  // namespace uniconn
