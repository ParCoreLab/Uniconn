

#include "../include/common.hpp"

typedef char real;

__global__ void comm_kernel(real *__restrict__ a_new, real *__restrict__ a, int nx, int neighbour,
                            uint64_t *__restrict__ signal_buf, uint64_t i, uniconn::Communicator<> *comm) {
    if (threadIdx.x == 0 && neighbour == 0) {
        uniconn::Coordinator<>::Acknowledge<uniconn::ThreadGroup::THREAD>(a_new, nx, signal_buf, i + 1, neighbour,
                                                                          comm);
    }
    uniconn::Coordinator<>::Post<uniconn::ThreadGroup::BLOCK>(a, a_new, nx, signal_buf, i + 1, neighbour, comm);

    if (threadIdx.x == 0 && neighbour == 1) {
        uniconn::Coordinator<>::Acknowledge<uniconn::ThreadGroup::THREAD>(a_new, nx, signal_buf, i + 1, neighbour,
                                                                          comm);
    }
}

int main(int argc, char *argv[]) {
    const int max_msg_size = get_argval<int>(argv, argv + argc, "-maxmsg", MAX_MESSAGE_SIZE);  // coll 1<<20

    long long unsigned int required_symmetric_heap_size = (2 * max_msg_size * sizeof(real) + 1 * sizeof(uint64_t)) *
                                                          1.1;  // Factor 2 is because 2 arrays are allocated - a and
                                                                // a_new 1.1 factor is just for alignment or other usage

    char symmetric_heap_size_str[100];
    sprintf(symmetric_heap_size_str, "%llu", required_symmetric_heap_size);
    setenv("NVSHMEM_SYMMETRIC_SIZE", symmetric_heap_size_str, 1);
    uniconn::Environment<> env(argc, argv);

    int rank = env.WorldRank();
    int size = env.WorldSize();
    int local_rank = env.NodeRank();
    // Add Local size and Rank
    env.SetDevice(local_rank);
    GPU_RT_CALL(UncGpuDeviceSynchronize());

    uniconn::Communicator<> comm;
    uniconn::Communicator<> *comm_d = comm.toDevice();

    size = comm.GlobalSize();
    rank = comm.GlobalRank();

    UncGpuStream_t stream;
    GPU_RT_CALL(UncGpuStreamCreate(&stream));

    UncGpuEvent_t startEvent, stopEvent;
    GPU_RT_CALL(UncGpuEventCreate(&startEvent));
    GPU_RT_CALL(UncGpuEventCreate(&stopEvent));

    // Memory allocation.
    real *send_buff = uniconn::Memory<>::Alloc<real>(max_msg_size);
    real *recv_buff = uniconn::Memory<>::Alloc<real>(max_msg_size);
    uint64_t *signal_buff = uniconn::Memory<>::Alloc<uint64_t>(1);
    const int neighbour = (rank + 1) % size;
    int message_size = 1;
    uint64_t i = 0;
    void *kernelArgsFull[] = {(void *)&recv_buff,   (void *)&send_buff, (void *)&message_size, (void *)&neighbour,
                              (void *)&signal_buff, (void *)&i,         (void *)&comm_d};

    uniconn::Coordinator<> send_coordinator(stream);

    send_coordinator.bindKernel<uniconn::LaunchMode::FullDevice>((void *)comm_kernel, dim3(1), dim3(1024), 0,
                                                                 kernelArgsFull);

    uniconn::Coordinator<> recv_coordinator(stream);

    recv_coordinator.bindKernel<uniconn::LaunchMode::FullDevice>((void *)comm_kernel, dim3(1), dim3(1024), 0,
                                                                 kernelArgsFull);

    int iter_max = LAT_LOOP_SMALL;
    uint64_t warmup = LAT_SKIP_SMALL;
    for (message_size = 0; message_size <= max_msg_size; message_size = (message_size ? message_size * 2 : 1)) {
        GPU_RT_CALL(UncGpuDeviceSynchronize());
        MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
        if (message_size > LARGE_MESSAGE_SIZE) {
            iter_max = LAT_LOOP_LARGE;
            warmup = LAT_SKIP_LARGE;
        }
        GPU_RT_CALL(UncGpuMemsetAsync(signal_buff, 0, 1 * sizeof(uint64_t), stream));
        comm.Barrier(stream);
        if (rank == 0) {
            double start = 0.0, end = 0.0;
            for (i = 0; i < warmup + iter_max; i++) {
                if (i == warmup) {
                    start = MPI_Wtime();
                    // GPU_RT_CALL(UncGpuEventRecord(startEvent, stream));
                }
                init_kernel<<<1, 1024, 0, stream>>>(recv_buff, send_buff, message_size);
                send_coordinator.LaunchKernel();
                send_coordinator.Post(send_buff, recv_buff, message_size, signal_buff, i + 1, neighbour, &comm);
                send_coordinator.Acknowledge(recv_buff, message_size, signal_buff, i + 1, neighbour, &comm);
            }
            GPU_RT_CALL(UncGpuStreamSynchronize(stream));
            end = MPI_Wtime();
            // GPU_RT_CALL(UncGpuEventRecord(stopEvent, stream));
            // GPU_RT_CALL(UncGpuEventSynchronize(stopEvent));
            // float multi_gpu_time = 0.0;
            // GPU_RT_CALL(UncGpuEventElapsedTime(&multi_gpu_time, startEvent, stopEvent));
            POP_RANGE
            printf("latency, unc, %d, %f\n", message_size, ((end - start) * 1e6) / (2.0 * iter_max));

        } else if (rank == 1)  // process 1
        {
            for (i = 0; i < warmup + iter_max; ++i) {
                recv_coordinator.LaunchKernel();
                recv_coordinator.Acknowledge(recv_buff, message_size, signal_buff, i + 1, neighbour, &comm);
                recv_coordinator.Post(send_buff, recv_buff, message_size, signal_buff, i + 1, neighbour, &comm);
            }
        }
    }

    GPU_RT_CALL(UncGpuDeviceSynchronize());
    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
    uniconn::Memory<>::Free(send_buff);
    uniconn::Memory<>::Free(recv_buff);
    uniconn::Memory<>::Free(signal_buff);
    GPU_RT_CALL(UncGpuEventDestroy(startEvent));
    GPU_RT_CALL(UncGpuEventDestroy(stopEvent));
    GPU_RT_CALL(UncGpuStreamDestroy(stream));

    return 0;
}