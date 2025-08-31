#include "common.hpp"
#include "uniconn/gpumpi/coordinator.hpp"
namespace uniconn {

// template <typename T>
// GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::Post(T* src_buffer, T* dest_buffer, size_t buffer_size,
//                                                                     uint64_t* signal_location, uint64_t signal_val,
//                                                                     int dest_process_id,
//                                                                     Communicator<MPIBackend>* comm) {
//     if (is_grouped == 0) {
//         if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
//             GPU_RT_CALL(UncGpuStreamSynchronize(stream));
//         }
//         MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Send)(src_buffer, buffer_size, internal::mpi::TypeMap<T>(),
//                                                     dest_process_id, 2, comm->mpi_comm));
//     } else {
//         MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Isend)(src_buffer, buffer_size, internal::mpi::TypeMap<T>(),
//                                                      dest_process_id, 2, comm->mpi_comm,
//                                                      &internal_reqs.emplace_back(MPI_REQUEST_NULL)));
//     }
// }

// template <typename T>
// GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::Acknowledge(T* dest_buffer, size_t buffer_size,
//                                                                            uint64_t* signal_location,
//                                                                            uint64_t signal_val, int src_process_id,
//                                                                            Communicator<MPIBackend>* comm) {
//     if (is_grouped == 0) {
//         if (UncGpuStreamQuery(stream) == UncGpuErrorNotReady) {
//             GPU_RT_CALL(UncGpuStreamSynchronize(stream));
//         }
//         MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Recv)(dest_buffer, buffer_size, internal::mpi::TypeMap<T>(),
//                                                     src_process_id, 2, comm->mpi_comm, MPI_STATUS_IGNORE));
//     } else {
//         MPI_CALL(UNC_MPI_LARGE_COUNT_CALL(MPI_Irecv)(dest_buffer, buffer_size, internal::mpi::TypeMap<T>(),
//                                                      src_process_id, 2, comm->mpi_comm,
//                                                      &internal_reqs.emplace_back(MPI_REQUEST_NULL)));
//     }
// }

// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Post(T* src_buffer, T* dest_buffer, size_t buffer_size,
//                                                                       uint64_t* signal_location, uint64_t signal_val,
//                                                                       int dest_process_id,
//                                                                       Communicator<MPIBackend>* comm) {}

// template <ThreadGroup SCOPE, typename T>
// GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Acknowledge(T* dest_buffer, size_t buffer_size,
//                                                                              uint64_t* signal_location,
//                                                                              uint64_t signal_val, int src_process_id,
//                                                                              Communicator<MPIBackend>* comm) {}

// #define EXPAND(TYPE)                                                                                               \
//     template GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::Post<TYPE>(                               \
//         TYPE * src_buffer, TYPE * dest_buffer, size_t buffer_size, uint64_t* signal_location, uint64_t signal_val,    \
//         int dest_process_id, Communicator<MPIBackend>* comm);                                                         \
//     template GPU_HOST void Coordinator<MPIBackend, LaunchMode::HostDriven>::Acknowledge<TYPE>(                        \
//         TYPE * dest_buffer, size_t buffer_size, uint64_t* signal_location, uint64_t signal_val, int src_process_id,   \
//         Communicator<MPIBackend>* comm);                                                                              \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Post<ThreadGroup::BLOCK, TYPE>(         \
//         TYPE * src_buffer, TYPE * dest_buffer, size_t buffer_size, uint64_t* signal_location, uint64_t signal_val,    \
//         int dest_process_id, Communicator<MPIBackend>* comm);                                                         \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Acknowledge<ThreadGroup::BLOCK, TYPE>(  \
//         TYPE * dest_buffer, size_t buffer_size, uint64_t* signal_location, uint64_t signal_val, int src_process_id,   \
//         Communicator<MPIBackend>* comm);                                                                              \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Post<ThreadGroup::WARP, TYPE>(          \
//         TYPE * src_buffer, TYPE * dest_buffer, size_t buffer_size, uint64_t* signal_location, uint64_t signal_val,    \
//         int dest_process_id, Communicator<MPIBackend>* comm);                                                         \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Acknowledge<ThreadGroup::WARP, TYPE>(   \
//         TYPE * dest_buffer, size_t buffer_size, uint64_t* signal_location, uint64_t signal_val, int src_process_id,   \
//         Communicator<MPIBackend>* comm);                                                                              \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Post<ThreadGroup::THREAD, TYPE>(        \
//         TYPE * src_buffer, TYPE * dest_buffer, size_t buffer_size, uint64_t* signal_location, uint64_t signal_val,    \
//         int dest_process_id, Communicator<MPIBackend>* comm);                                                         \
//     template GPU_DEVICE void Coordinator<MPIBackend, LaunchMode::HostDriven>::Acknowledge<ThreadGroup::THREAD, TYPE>( \
//         TYPE * dest_buffer, size_t buffer_size, uint64_t* signal_location, uint64_t signal_val, int src_process_id,   \
//         Communicator<MPIBackend>* comm);

// UNC_GPUMPI_REPT(EXPAND)

}  // namespace uniconn
