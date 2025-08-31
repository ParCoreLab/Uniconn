
#include "../include/common.hpp"

typedef char real;

__global__ void comm_kernel(real *__restrict__ const a_new, const real *__restrict__ const a, const int nx,
                            const int neighbour, uint64_t *__restrict__ const signal_buf, const uint64_t i) {
    if (threadIdx.x == 0 && neighbour == 0) {
        nvshmem_signal_wait_until(signal_buf, NVSHMEM_CMP_GE, i + 1);
    }

    nvshmemx_putmem_signal_nbi_block(a_new, a, nx * sizeof(real), signal_buf, 1, NVSHMEM_SIGNAL_ADD, neighbour);

    if (threadIdx.x == 0 && neighbour == 1) {
        nvshmem_signal_wait_until(signal_buf, NVSHMEM_CMP_GE, i + 1);
    }
}

int main(int argc, char *argv[]) {
    // Getting arguments from user.

    const int max_msg_size = get_argval<int>(argv, argv + argc, "-maxmsg", MAX_MESSAGE_SIZE);  // coll 1<<20

    // MPI initialization.
    MPI_CALL(MPI_Init(&argc, &argv));

    int rank;
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    int size;
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));
    int num_devices = 0;
    GPU_RT_CALL(UncGpuGetDeviceCount(&num_devices));

    // Divide the processors to their nodes.
    int local_rank = -1;
    int local_size = 1;
    {
        MPI_Comm local_comm;
        MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));
        MPI_CALL(MPI_Comm_rank(local_comm, &local_rank));
        MPI_CALL(MPI_Comm_free(&local_comm));
    }

    if (1 > num_devices || num_devices < local_size) {
        fprintf(stderr, "ERROR Number of visible devices (%d) is less than number of ranks on the node (%d)!\n",
                num_devices, local_size);
        MPI_CALL(MPI_Finalize());
        return 1;
    }

    GPU_RT_CALL(UncGpuSetDevice(local_rank % num_devices));

    // NVSHMEM initialization
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    attr.mpi_comm = &mpi_comm;

    long long unsigned int required_symmetric_heap_size = (2 * max_msg_size * sizeof(real) + 1 * sizeof(uint64_t)) *
                                                          1.1;  // Factor 2 is because 2 arrays are allocated - a and
                                                                // a_new 1.1 factor is just for alignment or other usage

    char symmetric_heap_size_str[100];
    sprintf(symmetric_heap_size_str, "%llu", required_symmetric_heap_size);
    setenv("NVSHMEM_SYMMETRIC_SIZE", symmetric_heap_size_str, 1);

    if (nvshmemx_init_status() == NVSHMEM_STATUS_NOT_INITIALIZED) {
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    }

    int npes = nvshmem_n_pes();
    int mype = nvshmem_my_pe();

    UncGpuStream_t nvshmem_stream;
    GPU_RT_CALL(UncGpuStreamCreate(&nvshmem_stream));

    UncGpuEvent_t startEvent, stopEvent;
    GPU_RT_CALL(UncGpuEventCreate(&startEvent));
    GPU_RT_CALL(UncGpuEventCreate(&stopEvent));

    // Memory allocation.
    real *send_buff;
    send_buff = (real *)nvshmem_malloc(max_msg_size * sizeof(real));
    real *recv_buff;
    recv_buff = (real *)nvshmem_malloc(max_msg_size * sizeof(real));
    uint64_t *signal_buff = (uint64_t *)nvshmem_malloc(sizeof(uint64_t));

    const int neighbour = (mype + 1) % npes;
    int message_size = 0;
    uint64_t i = 0;
    void *kernelArgs[] = {(void *)&recv_buff, (void *)&send_buff,   (void *)&message_size,
                          (void *)&neighbour, (void *)&signal_buff, (void *)&i};

    int iter_max = LAT_LOOP_SMALL;
    uint64_t warmup = LAT_SKIP_SMALL;
    for (message_size = 0; message_size <= max_msg_size; message_size = (message_size ? message_size * 2 : 1)) {
        if (message_size > LARGE_MESSAGE_SIZE) {
            iter_max = LAT_LOOP_LARGE;
            warmup = LAT_SKIP_LARGE;
        }
        GPU_RT_CALL(UncGpuMemsetAsync(signal_buff, 0, 1 * sizeof(uint64_t), nvshmem_stream));
        nvshmemx_barrier_all_on_stream(nvshmem_stream);
        GPU_RT_CALL(UncGpuDeviceSynchronize());
        if (mype == 0) {
            double start = 0.0, end = 0.0;
            for (i = 0; i < warmup + iter_max; i++) {
                if (i == warmup) {
                    start = MPI_Wtime();
                    // GPU_RT_CALL(UncGpuEventRecord(startEvent, nvshmem_stream));
                }
                init_kernel<<<1, 1024, 0, nvshmem_stream>>>(recv_buff, send_buff, message_size);
                nvshmemx_collective_launch((void *)comm_kernel, dim3(1), dim3(1024), kernelArgs, 0, nvshmem_stream);
            }
            GPU_RT_CALL(UncGpuStreamSynchronize(nvshmem_stream));
            end = MPI_Wtime();
            // GPU_RT_CALL(UncGpuEventRecord(stopEvent, nvshmem_stream));
            // GPU_RT_CALL(UncGpuEventSynchronize(stopEvent));
            // float multi_gpu_time = 0.0;
            // GPU_RT_CALL(UncGpuEventElapsedTime(&multi_gpu_time, startEvent, stopEvent));
            POP_RANGE
            printf("latency, nvshmem_d, %d, %f\n", message_size, ((end - start) * 1e6) / (2.0 * iter_max));

        } else if (mype == 1)  // process 1
        {
            for (i = 0; i < warmup + iter_max; ++i) {
                nvshmemx_collective_launch((void *)comm_kernel, dim3(1), dim3(1024), kernelArgs, 0, nvshmem_stream);
            }
        }
    }

    GPU_RT_CALL(UncGpuDeviceSynchronize());
    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
    nvshmem_free(send_buff);
    nvshmem_free(recv_buff);
    nvshmem_free(signal_buff);
    GPU_RT_CALL(UncGpuEventDestroy(startEvent));
    GPU_RT_CALL(UncGpuEventDestroy(stopEvent));
    GPU_RT_CALL(UncGpuStreamDestroy(nvshmem_stream));
    nvshmem_finalize();
    MPI_CALL(MPI_Finalize());
    return 0;
}