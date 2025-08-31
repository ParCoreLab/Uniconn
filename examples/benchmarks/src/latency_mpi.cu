
#include "../include/common.hpp"

typedef char real;
#define MPI_REAL_TYPE MPI_BYTE

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

    UncGpuStream_t mpi_stream;
    GPU_RT_CALL(UncGpuStreamCreate(&mpi_stream));

    UncGpuEvent_t startEvent, stopEvent;
    GPU_RT_CALL(UncGpuEventCreate(&startEvent));
    GPU_RT_CALL(UncGpuEventCreate(&stopEvent));

    // Memory allocation.
    real *send_buff;
    GPU_RT_CALL(UncGpuMalloc(&send_buff, max_msg_size * sizeof(real)));
    real *recv_buff;
    GPU_RT_CALL(UncGpuMalloc(&recv_buff, max_msg_size * sizeof(real)));

    MPI_Status reqstat;
    // Iterate over message sizes.
    const int neighbour = (rank + 1) % size;

    int iter_max = LAT_LOOP_SMALL;
    uint64_t warmup = LAT_SKIP_SMALL;

    for (int message_size = 0; message_size <= max_msg_size; message_size = (message_size ? message_size * 2 : 1)) {
        if (message_size > LARGE_MESSAGE_SIZE) {
            iter_max = LAT_LOOP_LARGE;
            warmup = LAT_SKIP_LARGE;
        }
        GPU_RT_CALL(UncGpuDeviceSynchronize());
        MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
        if (rank == 0) {
            double start = 0.0, end = 0.0;
            for (uint64_t i = 0; i < warmup + iter_max; i++) {
                if (i == warmup) {
                    start = MPI_Wtime();
                    // GPU_RT_CALL(UncGpuEventRecord(startEvent, mpi_stream));
                }
                init_kernel<<<1, 1024, 0, mpi_stream>>>(recv_buff, send_buff, message_size);
                GPU_RT_CALL(UncGpuStreamSynchronize(mpi_stream));
                MPI_CALL(MPI_Send(send_buff, message_size, MPI_REAL_TYPE, neighbour, 1, MPI_COMM_WORLD));
                MPI_CALL(MPI_Recv(recv_buff, message_size, MPI_REAL_TYPE, neighbour, 1, MPI_COMM_WORLD, &reqstat));
            }
            GPU_RT_CALL(UncGpuStreamSynchronize(mpi_stream));
            end = MPI_Wtime();
            // GPU_RT_CALL(UncGpuEventRecord(stopEvent, mpi_stream));
            // GPU_RT_CALL(UncGpuEventSynchronize(stopEvent));
            // float multi_gpu_time = 0.0;
            // GPU_RT_CALL(UncGpuEventElapsedTime(&multi_gpu_time, startEvent, stopEvent));
            POP_RANGE
            printf("latency, mpi, %d, %f\n", message_size, ((end - start) * 1e6) / (2.0 * iter_max));

        } else if (rank == 1)  // process 1
        {
            for (int i = 0; i < warmup + iter_max; ++i) {
                MPI_CALL(MPI_Recv(recv_buff, message_size, MPI_REAL_TYPE, neighbour, 1, MPI_COMM_WORLD, &reqstat));
                MPI_CALL(MPI_Send(send_buff, message_size, MPI_REAL_TYPE, neighbour, 1, MPI_COMM_WORLD));
            }
        }
    }

    GPU_RT_CALL(UncGpuDeviceSynchronize());
    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
    GPU_RT_CALL(UncGpuFree(recv_buff));
    GPU_RT_CALL(UncGpuFree(send_buff));
    GPU_RT_CALL(UncGpuEventDestroy(startEvent));
    GPU_RT_CALL(UncGpuEventDestroy(stopEvent));
    GPU_RT_CALL(UncGpuStreamDestroy(mpi_stream));
    MPI_CALL(MPI_Finalize());
    return 0;
}
